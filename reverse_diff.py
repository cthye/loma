import ir

ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
import autodiff
import string
import random
import math


# From https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits
def random_id_generator(
    size=6, chars=string.ascii_lowercase + string.ascii_uppercase + string.digits
):
    return "".join(random.choice(chars) for _ in range(size))


def reverse_diff(
    diff_func_id: str,
    structs: dict[str, loma_ir.Struct],
    funcs: dict[str, loma_ir.func],
    diff_structs: dict[str, loma_ir.Struct],
    func: loma_ir.FunctionDef,
    func_to_rev: dict[str, str],
) -> loma_ir.FunctionDef:
    """Given a primal loma function func, apply reverse differentiation
    and return a function that computes the total derivative of func.

    For example, given the following function:
    def square(x : In[float]) -> float:
        return x * x
    and let diff_func_id = 'd_square', reverse_diff() should return
    def d_square(x : In[float], _dx : Out[float], _dreturn : float):
        _dx = _dx + _dreturn * x + _dreturn * x

    Parameters:
    diff_func_id - the ID of the returned function
    structs - a dictionary that maps the ID of a Struct to
            the corresponding Struct
    funcs - a dictionary that maps the ID of a function to
            the corresponding func
    diff_structs - a dictionary that maps the ID of the primal
            Struct to the corresponding differential Struct
            e.g., diff_structs['float'] returns _dfloat
    func - the function to be differentiated
    func_to_rev - mapping from primal function ID to its reverse differentiation
    """

    # Some utility functions you can use for your homework.
    def type_to_string(t):
        match t:
            case loma_ir.Int():
                return "int"
            case loma_ir.Float():
                return "float"
            case loma_ir.Array():
                return "array_" + type_to_string(t.t)
            case loma_ir.Struct():
                return t.id
            case _:
                assert False

    def assign_zero(target):
        match target.t:
            case loma_ir.Int():
                return []
            case loma_ir.Float():
                return [loma_ir.Assign(target, loma_ir.ConstFloat(0.0))]
            case loma_ir.Struct():
                s = target.t
                stmts = []
                for m in s.members:
                    target_m = loma_ir.StructAccess(target, m.id, t=m.t)
                    if isinstance(m.t, loma_ir.Float):
                        stmts += assign_zero(target_m)
                    elif isinstance(m.t, loma_ir.Int):
                        pass
                    elif isinstance(m.t, loma_ir.Struct):
                        stmts += assign_zero(target_m)
                    else:
                        assert isinstance(m.t, loma_ir.Array)
                        assert m.t.static_size is not None
                        for i in range(m.t.static_size):
                            target_m = loma_ir.ArrayAccess(
                                target_m, loma_ir.ConstInt(i), t=m.t.t
                            )
                            stmts += assign_zero(target_m)
                return stmts
            case _:
                assert (
                    False
                ), f"Reverse mode assign_zero error: unhandled target type {target.t}"

    def accum_deriv(target, deriv, overwrite):
        # print(f"debug target {target}")
        # print(f"debug target type {target.t}")
        match target.t:
            case loma_ir.Int():
                return []
            case loma_ir.Float():
                if overwrite:
                    return [loma_ir.Assign(target, deriv)]
                else:
                    return [
                        loma_ir.Assign(
                            target, loma_ir.BinaryOp(loma_ir.Add(), target, deriv)
                        )
                    ]
            case loma_ir.Struct():
                s = target.t
                stmts = []
                for m in s.members:
                    target_m = loma_ir.StructAccess(target, m.id, t=m.t)
                    deriv_m = loma_ir.StructAccess(deriv, m.id, t=m.t)
                    if isinstance(m.t, loma_ir.Float):
                        stmts += accum_deriv(target_m, deriv_m, overwrite)
                    elif isinstance(m.t, loma_ir.Int):
                        pass
                    elif isinstance(m.t, loma_ir.Struct):
                        stmts += accum_deriv(target_m, deriv_m, overwrite)
                    else:
                        assert isinstance(m.t, loma_ir.Array)
                        assert m.t.static_size is not None
                        for i in range(m.t.static_size):
                            target_m = loma_ir.ArrayAccess(
                                target_m, loma_ir.ConstInt(i), t=m.t.t
                            )
                            deriv_m = loma_ir.ArrayAccess(
                                deriv_m, loma_ir.ConstInt(i), t=m.t.t
                            )
                            stmts += accum_deriv(target_m, deriv_m, overwrite)
                return stmts
            case _:
                assert False

    def check_lhs_is_output_arg(lhs, output_arg_ids):
        match lhs:
            case loma_ir.Var():
                return lhs.id in output_arg_ids
            case loma_ir.StructAccess():
                return check_lhs_is_output_arg(lhs.struct, output_arg_ids)
            case loma_ir.ArrayAccess():
                return check_lhs_is_output_arg(lhs.array, output_arg_ids)
            case _:
                assert (
                    False
                ), f"Reverse mode check_lhs_is_output_arg error: unhandled lhs {lhs}"

    def assign_struct(struct_acc, val):
        """
        construct a new StructAccess expr that access a new struct with val assigned

        struct_acc: a StructAccess expr
        val: loma_ir.Var
        return a struct
        """
        new_struct = struct_acc
        match struct_acc.struct:
            case loma_ir.Var():
                new_struct.struct = val
                return new_struct
            case loma_ir.StructAccess():
                # for nested struct access
                return loma_ir.StructAccess(
                    struct=assign_struct(struct_acc.struct, val),
                    member_id=struct_acc.member_id,
                    lineno=struct_acc.lineno,
                    t=struct_acc.t,
                )
            case _:
                assert (
                    False
                ), f"Reverse mode assign_struct error: unhandled struct type {struct_acc.struct}"

    # A utility class that you can use for HW3.
    # This mutator normalizes each call expression into
    # f(x0, x1, ...)
    # where x0, x1, ... are all loma_ir.Var or
    # loma_ir.ArrayAccess or loma_ir.StructAccess
    class CallNormalizeMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            self.tmp_count = 0
            self.tmp_declare_stmts = []
            new_body = [self.mutate_stmt(stmt) for stmt in node.body]
            new_body = irmutator.flatten(new_body)

            new_body = self.tmp_declare_stmts + new_body

            return loma_ir.FunctionDef(
                node.id,
                node.args,
                new_body,
                node.is_simd,
                node.ret_type,
                lineno=node.lineno,
            )

        def mutate_return(self, node):
            self.tmp_assign_stmts = []
            val = self.mutate_expr(node.val)
            return self.tmp_assign_stmts + [loma_ir.Return(val, lineno=node.lineno)]

        def mutate_declare(self, node):
            self.tmp_assign_stmts = []
            val = None
            if node.val is not None:
                val = self.mutate_expr(node.val)
            return self.tmp_assign_stmts + [
                loma_ir.Declare(node.target, node.t, val, lineno=node.lineno)
            ]

        def mutate_assign(self, node):
            self.tmp_assign_stmts = []
            target = self.mutate_expr(node.target)
            val = self.mutate_expr(node.val)
            return self.tmp_assign_stmts + [
                loma_ir.Assign(target, val, lineno=node.lineno)
            ]

        def mutate_call_stmt(self, node):
            self.tmp_assign_stmts = []
            call = self.mutate_expr(node.call)
            return self.tmp_assign_stmts + [loma_ir.CallStmt(call, lineno=node.lineno)]

        def mutate_call(self, node):
            new_args = []
            for arg in node.args:
                if (
                    not isinstance(arg, loma_ir.Var)
                    and not isinstance(arg, loma_ir.ArrayAccess)
                    and not isinstance(arg, loma_ir.StructAccess)
                ):
                    arg = self.mutate_expr(arg)
                    tmp_name = f"_call_t_{self.tmp_count}_{random_id_generator()}"
                    self.tmp_count += 1
                    tmp_var = loma_ir.Var(tmp_name, t=arg.t)
                    self.tmp_declare_stmts.append(loma_ir.Declare(tmp_name, arg.t))
                    self.tmp_assign_stmts.append(loma_ir.Assign(tmp_var, arg))
                    new_args.append(tmp_var)
                else:
                    new_args.append(arg)
            return loma_ir.Call(node.id, new_args, t=node.t)

    # HW2 happens here. Modify the following IR mutators to perform
    # reverse differentiation.
    class FwdPassMutator(irmutator.IRMutator):
        # def mutate_function_def(self, node):
        def mutate_function_def(self, node):
            self.fwd_stmts = []
            self.assign_array_dict = {}
            self.output_arg_ids = []

            for arg in node.args:
                if arg.i == loma_ir.Out():
                    self.output_arg_ids.append(arg.id)

            for stmt in node.body:
                self.mutate_stmt(stmt)

            # declare arrays for assign statements' stacks
            array_stmts = []
            # print(f"??? {self.assign_array_dict.items()}")z/x

            for array_t, cnt in self.assign_array_dict.items():
                # get the origin type from array name
                array_name = f"_t_{type_to_string(array_t)}"

                array_stmts.append(
                    loma_ir.Declare(
                        target=array_name, t=loma_ir.Array(t=array_t, static_size=cnt)
                    )
                )
                array_stmts.append(
                    loma_ir.Declare(
                        target=f"_stack_ptr_{type_to_string(array_t)}", t=loma_ir.Int()
                    )
                )

            self.fwd_stmts = array_stmts + self.fwd_stmts

        # def mutate_stmt(self, node):
        #     pass

        def mutate_declare(self, node):
            self.fwd_stmts.append(node)

            if isinstance(node.t, loma_ir.Int):  # ignore the ajoint of a int variable
                return
            ajoint_declare = loma_ir.Declare(
                target=f"_d{node.target}", t=node.t, lineno=node.lineno
            )
            self.fwd_stmts.append(ajoint_declare)

            # zero the struct?
            if isinstance(node.t, loma_ir.Struct):
                self.fwd_stmts += assign_zero(
                        loma_ir.Var(id=f"_d{node.target}", t=node.t, lineno=node.lineno)
                    )
                

        def mutate_assign(self, node):
            # ignore if lhs is output argument
            if check_lhs_is_output_arg(node.target, self.output_arg_ids):
                return

            array_name = f"_t_{type_to_string(node.target.t)}"
            # if array_name in self.assign_array_dict:
            #     self.assign_array_dict[array_name] += 1
            # else:
            #     self.assign_array_dict[array_name] = 0
            if node.target.t in self.assign_array_dict:
                self.assign_array_dict[node.target.t] += 1
            else:
                self.assign_array_dict[node.target.t] = 1

            # push the target variable into array
            stack_ptr = f"_stack_ptr_{type_to_string(node.target.t)}"
            push_assign_stmt = loma_ir.Assign(
                target=loma_ir.ArrayAccess(
                    array=loma_ir.Var(id=array_name), index=loma_ir.Var(id=stack_ptr)
                ),
                val=node.target,
            )

            push_move_ptr_stmt = loma_ir.Assign(
                target=loma_ir.Var(id=stack_ptr),
                val=loma_ir.BinaryOp(
                    op=loma_ir.Add(),
                    left=loma_ir.Var(id=stack_ptr),
                    right=loma_ir.ConstInt(1),
                ),
            )

            self.fwd_stmts = self.fwd_stmts + [
                push_assign_stmt,
                push_move_ptr_stmt,
                node,
            ]

    # Apply the differentiation.
    class RevDiffMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            # HW2: TODO
            # return super().mutate_function_def(node)
            self.global_temporary_ajoint_var_names = {}
            self.output_arg_ids = []

            fm = FwdPassMutator()
            fm.mutate_function_def(node)
            fwd_body = fm.fwd_stmts

            # print(fwd_body)

            # handle the signature
            new_args = []
            for arg in node.args:
                if arg.i == loma_ir.In():
                    new_args.append(arg)
                    # TODO: unique name for args
                    new_args.append(
                        loma_ir.Arg(id=f"_d{arg.id}", t=arg.t, i=loma_ir.Out())
                    )
                elif arg.i == loma_ir.Out():
                    new_args.append(
                        loma_ir.Arg(id=f"_d{arg.id}", t=arg.t, i=loma_ir.In())
                    )
                    self.output_arg_ids.append(arg.id)

            # TODO: unique name for return
            if node.ret_type != None:
                new_args.append(
                    loma_ir.Arg(
                        id="_dreturn",
                        t=node.ret_type,
                        i=loma_ir.In(),
                    )
                )

            rev_body = [self.mutate_stmt(stmt) for stmt in reversed(node.body)]
            # Important: mutate_stmt can return a list of statements. We need to flatten the list.
            rev_body = irmutator.flatten(rev_body)

            new_body = fwd_body + rev_body
            return loma_ir.FunctionDef(
                diff_func_id, new_args, new_body, node.is_simd, None, lineno=node.lineno
            )

        def mutate_return(self, node):
            # HW2: TODO
            # return super().mutate_return(node)

            # return x (_dreturn = x)
            # =>
            # _dx += _dreturn
            self.ajoint = loma_ir.Var(id="_dreturn", t=node.val.t, lineno=node.lineno)
            stmts = self.mutate_expr(node.val)
            self.ajoint = None
            return stmts

        def mutate_declare(self, node):
            # HW2: TODO
            # return super().mutate_declare(node)
            if node.val is not None:
                if isinstance(node.t, loma_ir.Int):
                    return []
                self.ajoint = loma_ir.Var(id=f"_d{node.target}")
                stmts = self.mutate_expr(node.val)
                self.ajoint = None
                return stmts
            return []

        def mutate_assign(self, node):
            # HW2: TODO
            # return super().mutate_assign(node)

            # if lhs is output arg, ignore the assign cos we don't have derivative of output arg
            if check_lhs_is_output_arg(node.target, self.output_arg_ids):
                if isinstance(node.target, loma_ir.Var):
                    self.ajoint = loma_ir.Var(id=f"_d{node.target.id}")
                elif isinstance(node.target, loma_ir.ArrayAccess):
                    self.ajoint = self.mutate_array_helper(node.target)
                elif isinstance(node.target, loma_ir.StructAccess):
                    # TODO: testcase?
                    new_struct_access = self.mutate_struct_helper(node.target)
                    stmts += assign_zero(new_struct_access)
                    self.ajoint = new_struct_access
                else:
                    assert (
                        False
                    ), f"Reverse mode mutate_assign error: unhandled node target type {node.target}"
                stmts = self.mutate_expr(node.val)
                self.ajoint = None
                return stmts

            stmts = []
            # recover the value from stack
            stack_ptr = f"_stack_ptr_{type_to_string(node.target.t)}"
            stmts.append(
                loma_ir.Assign(
                    target=loma_ir.Var(id=stack_ptr),
                    val=loma_ir.BinaryOp(
                        op=loma_ir.Sub(),
                        left=loma_ir.Var(id=stack_ptr),
                        right=loma_ir.ConstInt(1),
                    ),
                )
            )

            array_name = f"_t_{type_to_string(node.target.t)}"

            stmts.append(
                loma_ir.Assign(
                    target=node.target,
                    val=loma_ir.ArrayAccess(
                        array=loma_ir.Var(id=array_name),
                        index=loma_ir.Var(id=stack_ptr),
                    ),
                )
            )

            # compute the adjoints of the inputs
            if isinstance(node.target, loma_ir.Var):
                self.ajoint = loma_ir.Var(id=f"_d{node.target.id}")
            elif isinstance(node.target, loma_ir.StructAccess):
                new_struct_access = self.mutate_struct_helper(node.target)
                self.ajoint = new_struct_access
            else:
                assert (
                    False
                ), f"Reverse mode mutate_assign error: unhandled node target {node.target}"

            val_expr = self.mutate_expr(node.val)
            self.ajoint = None

            adjoint_vars = []
            for i in range(len(val_expr)):
                assert isinstance(val_expr[i], loma_ir.Assign)
                ori_target = val_expr[i].target
                ori_val = val_expr[i].val

                # subtract the derivative of input from ori_val
                new_val = loma_ir.BinaryOp(
                    loma_ir.Sub(),
                    ori_val,
                    ori_target,
                    t=ori_target.t,
                    lineno=node.lineno,
                )

                adjoint_var_name = f"_adj_{i}"
                adjoint_var = loma_ir.Var(
                    id=adjoint_var_name, t=ori_target.t, lineno=node.lineno
                )
                adjoint_vars.append(adjoint_var)

                if adjoint_var_name not in self.global_temporary_ajoint_var_names:
                    # declare first if the _adj_i var doesn't exist
                    declare_ajoint_stmt = loma_ir.Declare(
                        target=adjoint_var_name, t=ori_target.t, lineno=node.lineno
                    )
                    stmts.append(declare_ajoint_stmt)
                    self.global_temporary_ajoint_var_names[adjoint_var_name] = True

                assign_ajoint_stmt = loma_ir.Assign(
                    target=adjoint_var, val=new_val, lineno=node.lineno
                )
                stmts.append(assign_ajoint_stmt)

            # zero the differential of the target
            if isinstance(node.target, loma_ir.Var):
                stmts = stmts + assign_zero(
                    loma_ir.Var(
                        id=f"_d{node.target.id}",
                        t=node.target.t,
                        lineno=node.target.lineno,
                    )
                )
            elif isinstance(node.target, loma_ir.StructAccess):
                new_struct_access = self.mutate_struct_helper(node.target)
                stmts += assign_zero(new_struct_access)
            else:
                assert (
                    False
                ), f"Reverse mode mutate_assign error: unhandled node target {node.target}"

            # accumulate the ajoints to input's differentials
            for i in range(len(val_expr)):
                accum_stmt = accum_deriv(
                    target=val_expr[i].target, deriv=adjoint_vars[i], overwrite=False
                )
                stmts.append(accum_stmt)

            return stmts

        def mutate_ifelse(self, node):
            # HW3: TODO
            return super().mutate_ifelse(node)

        def mutate_call_stmt(self, node):
            # HW3: TODO
            return super().mutate_call_stmt(node)

        def mutate_while(self, node):
            # HW3: TODO
            return super().mutate_while(node)

        def mutate_const_float(self, node):
            # HW2: TODO
            # return super().mutate_const_float(node)
            return []

        def mutate_const_int(self, node):
            # HW2: TODO
            # return super().mutate_const_int(node)
            return []

        def mutate_var(self, node):
            # HW2: TODO
            # return super().mutate_var(node)

            # dx += ajoint
            new_target = loma_ir.Var(
                id="_d" + node.id,
                t=node.t,
                lineno=node.lineno,
            )
            if isinstance(node.t, loma_ir.Struct):
                stmts = accum_deriv(new_target, self.ajoint, False)
                print(f"debug stmts {stmts}")
                return stmts
            
            new_val = loma_ir.BinaryOp(
                op=loma_ir.Add(),
                left=new_target,
                right=self.ajoint,
                lineno=node.lineno,
                t=node.t,
            )
            return [loma_ir.Assign(target=new_target, val=new_val, lineno=node.lineno)]

        def mutate_array_helper(self, node):
            """
            return the derivative version of ArrayAccess expr
            """
            if isinstance(node.array, loma_ir.Var):
                return loma_ir.ArrayAccess(
                    array=loma_ir.Var(
                        id=f"_d{node.array.id}",
                        lineno=node.array.lineno,
                        t=node.array.t,
                    ),
                    index=node.index,
                    lineno=node.lineno,
                    t=node.t,
                )
            return loma_ir.ArrayAccess(
                array=self.mutate_array_helper(node.array),
                index=node.index,
                lineno=node.lineno,
                t=node.t,
            )

        def mutate_struct_helper(self, node):
            """
            return the derivative version of StructAccess expr
            """
            if isinstance(node.struct, loma_ir.Var):
                return loma_ir.StructAccess(
                    struct=loma_ir.Var(
                        id=f"_d{node.struct.id}",
                        lineno=node.struct.lineno,
                        t=node.struct.t,
                    ),
                    member_id=node.member_id,
                    lineno=node.lineno,
                    t=node.t,
                )
            return loma_ir.StructAccess(
                struct=self.mutate_struct_helper(node.struct),
                member_id=node.member_id,
                lineno=node.lineno,
                t=node.t,
            )

        def mutate_array_access(self, node):
            # HW2: TODO

            new_target = self.mutate_array_helper(node)

            new_val = loma_ir.BinaryOp(
                op=loma_ir.Add(),
                left=new_target,
                right=self.ajoint,
                lineno=node.lineno,
                t=node.t,
            )
            return [loma_ir.Assign(target=new_target, val=new_val, lineno=node.lineno)]

        def mutate_struct_access(self, node):
            # HW2: TODO
            # return super().mutate_struct_access(node)
            stmts = []

            # zero the derivative of struct i.e. dFoo
            new_struct_access = self.mutate_struct_helper(node)
            # stmts += assign_zero(new_struct_access)

            stmts += accum_deriv(new_struct_access, self.ajoint, False)
            return stmts

        def mutate_add(self, node):
            # HW2: TODO
            # return super().mutate_add(node)

            # f(x, y) = x + y
            # =>
            # dTf(x, _dx, y, _dy, _dout)
            # [
            #    _dx += _dout
            #    _dy += _dout
            # ]
            left = self.mutate_expr(node.left)
            right = self.mutate_expr(node.right)
            return left + right

        def mutate_sub(self, node):
            # HW2: TODO
            # return super().mutate_sub(node)

            # f(x, y) = x - y
            # =>
            # dTf(x, _dx, y, _dy, _dout)
            # [
            #    _dx += _dout
            #    _dy += -_dout
            # ]
            old_ajoint = self.ajoint
            left = self.mutate_expr(node.left)

            self.ajoint = loma_ir.BinaryOp(
                loma_ir.Sub(), left=loma_ir.ConstFloat(0.0), right=self.ajoint
            )
            right = self.mutate_expr(node.right)

            self.ajoint = old_ajoint
            return left + right

        def mutate_mul(self, node):
            # HW2: TODO
            # return super().mutate_mul(node)

            # f(x, y) = x * y
            # =>
            # dTf(x, _dx, y, _dy, _dout)
            # [
            #    _dx += _dout * y
            #    _dy += _dout * x
            # ]

            old_ajoint = self.ajoint
            self.ajoint = loma_ir.BinaryOp(loma_ir.Mul(), self.ajoint, node.right)
            left = self.mutate_expr(node.left)

            self.ajoint = old_ajoint
            self.ajoint = loma_ir.BinaryOp(loma_ir.Mul(), self.ajoint, node.left)
            right = self.mutate_expr(node.right)

            self.ajoint = old_ajoint
            return left + right

        def mutate_div(self, node):
            # HW2: TODO
            # return super().mutate_div(node)
            # f(x, y) = x / y
            # =>
            # dTf(x, _dx, y, _dy, _dout)
            # [
            #    _dx += _dout / y
            #    _dy += -_dout * x / y^2
            # ]

            old_ajoint = self.ajoint
            self.ajoint = loma_ir.BinaryOp(loma_ir.Div(), self.ajoint, node.right)
            left = self.mutate_expr(node.left)

            self.ajoint = old_ajoint
            self.ajoint = loma_ir.BinaryOp(
                loma_ir.Div(),
                loma_ir.BinaryOp(
                    loma_ir.Div(),
                    loma_ir.BinaryOp(
                        loma_ir.Sub(),
                        loma_ir.ConstFloat(0.0),
                        loma_ir.BinaryOp(loma_ir.Mul(), self.ajoint, node.left),
                    ),
                    node.right,
                ),
                node.right,
            )
            right = self.mutate_expr(node.right)

            self.ajoint = old_ajoint
            return left + right

        def mutate_call(self, node):
            # HW2: TODO
            # return super().mutate_call(node)
            arg_vals = []
            for arg in node.args:
                arg_vals.append(arg)

            match node.id:
                case "sin":
                    # f(x, y) = sin(x)
                    # =>
                    # dTf(x, _dx, _dout)
                    # [
                    #    _dx += _dout * cos(x)
                    # ]
                    old_ajoint = self.ajoint
                    self.ajoint = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        self.ajoint,
                        loma_ir.Call(
                            "cos",
                            arg_vals,
                            lineno=node.lineno,
                            t=node.t,
                        ),
                    )
                    stmt = self.mutate_expr(arg_vals[0])
                    self.ajoint = old_ajoint
                    return stmt
                case "cos":
                    # f(x, y) = cos(x)
                    # =>
                    # dTf(x, _dx, _dout)
                    # [
                    #    _dx += _dout * -sin(x)
                    # ]
                    old_ajoint = self.ajoint
                    self.ajoint = loma_ir.BinaryOp(
                        loma_ir.Sub(),
                        loma_ir.ConstFloat(0.0),
                        loma_ir.BinaryOp(
                            loma_ir.Mul(),
                            self.ajoint,
                            loma_ir.Call(
                                "sin",
                                arg_vals,
                                lineno=node.lineno,
                                t=node.t,
                            ),
                        ),
                    )
                    stmt = self.mutate_expr(arg_vals[0])
                    self.ajoint = old_ajoint
                    return stmt
                case "sqrt":
                    # f(x, y) = sqrt(x)
                    # =>
                    # dTf(x, _dx, _dout)
                    # [
                    #    _dx += _dout * 0.5 / sqrt(x)
                    # ]
                    old_ajoint = self.ajoint
                    self.ajoint = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        loma_ir.ConstFloat(0.5),
                        loma_ir.BinaryOp(
                            loma_ir.Div(),
                            self.ajoint,
                            loma_ir.Call(
                                node.id,
                                arg_vals,
                                lineno=node.lineno,
                                t=node.t,
                            ),
                        ),
                    )
                    stmt = self.mutate_expr(arg_vals[0])
                    self.ajoint = old_ajoint
                    return stmt
                case "pow":
                    # f(x, y) = pow(x, y)
                    # =>
                    # dTf(x, _dx, y, _dy, _dout)
                    # [
                    #    _dx += _dout * y * x^{y-1}
                    #    _dy += _dout * x^y * log(x)
                    # ]
                    x = arg_vals[0]
                    y = arg_vals[1]

                    old_ajoint = self.ajoint
                    self.ajoint = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        loma_ir.Call(
                            "pow",
                            [
                                x,
                                loma_ir.BinaryOp(
                                    loma_ir.Sub(),
                                    y,
                                    loma_ir.ConstInt(1),
                                    lineno=node.lineno,
                                    t=node.t,
                                ),
                            ],
                            lineno=node.lineno,
                            t=node.t,
                        ),
                        loma_ir.BinaryOp(
                            loma_ir.Mul(),
                            self.ajoint,
                            y,
                        ),
                    )
                    stmt_1 = self.mutate_expr(arg_vals[0])
                    self.ajoint = old_ajoint

                    self.ajoint = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        loma_ir.Call("log", [x], lineno=node.lineno, t=node.t),
                        loma_ir.BinaryOp(
                            loma_ir.Mul(),
                            self.ajoint,
                            loma_ir.Call(node.id, [x, y], lineno=node.lineno, t=node.t),
                        ),
                    )
                    stmt_2 = self.mutate_expr(arg_vals[1])
                    self.ajoint = old_ajoint

                    return stmt_1 + stmt_2
                case "exp":
                    # f(x, y) = exp(x)
                    # =>
                    # dTf(x, _dx, _dout)
                    # [
                    #    _dx += _dout * exp(x)
                    # ]
                    old_ajoint = self.ajoint
                    self.ajoint = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        self.ajoint,
                        loma_ir.Call(node.id, arg_vals, lineno=node.lineno, t=node.t),
                    )
                    stmt = self.mutate_expr(arg_vals[0])
                    self.ajoint = old_ajoint
                    return stmt
                case "log":
                    # f(x, y) = log(x)
                    # =>
                    # dTf(x, _dx, _dout)
                    # [
                    #    _dx += _dout / x
                    # ]
                    old_ajoint = self.ajoint
                    self.ajoint = loma_ir.BinaryOp(
                        loma_ir.Div(),
                        self.ajoint,
                        arg_vals[0],
                    )
                    stmt = self.mutate_expr(arg_vals[0])
                    self.ajoint = old_ajoint
                    return stmt
                case "float2int":
                    # f(x, y) = float2int(x)
                    # =>
                    # dTf(x, _dx, _dout)
                    # [
                    #    _dx += 0
                    # ]
                    return []
                case "int2float":
                    return []
                case _:
                    assert (
                        False
                    ), f"Reverse mode mutate_call error: unhandled call id {node.id}"

    return RevDiffMutator().mutate_function_def(func)
