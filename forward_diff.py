import ir

ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
import autodiff


def forward_diff(
    diff_func_id: str,
    structs: dict[str, loma_ir.Struct],
    funcs: dict[str, loma_ir.func],
    diff_structs: dict[str, loma_ir.Struct],
    func: loma_ir.FunctionDef,
    func_to_fwd: dict[str, str],
) -> loma_ir.FunctionDef:
    """Given a primal loma function func, apply forward differentiation
    and return a function that computes the total derivative of func.

    For example, given the following function:
    def square(x : In[float]) -> float:
        return x * x
    and let diff_func_id = 'd_square', forward_diff() should return
    def d_square(x : In[_dfloat]) -> _dfloat:
        return make__dfloat(x.val * x.val, x.val * x.dval + x.dval * x.val)
    where the class _dfloat is
    class _dfloat:
        val : float
        dval : float
    and the function make__dfloat is
    def make__dfloat(val : In[float], dval : In[float]) -> _dfloat:
        ret : _dfloat
        ret.val = val
        ret.dval = dval
        return ret

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
    func_to_fwd - mapping from primal function ID to its forward differentiation
    """

    # HW1 happens here. Modify the following IR mutators to perform
    # forward differentiation.

    # Apply the differentiation.
    class FwdDiffMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            # HW1: TODO
            # return super().mutate_function_def(node)
            diff_args = []
            for arg in node.args:
                dArgType = autodiff.type_to_diff_type(diff_structs, arg.t)
                diff_args.append(loma_ir.Arg(arg.id, dArgType, arg.i))
            diff_return = autodiff.type_to_diff_type(diff_structs, node.ret_type)

            new_body = [self.mutate_stmt(stmt) for stmt in node.body]
            # Important: mutate_stmt can return a list of statements. We need to flatten the list.
            new_body = irmutator.flatten(new_body)

            return loma_ir.FunctionDef(
                diff_func_id,
                diff_args,
                new_body,
                node.is_simd,
                diff_return,
                lineno=node.lineno,
            )

        def mutate_return(self, node):
            # HW1: TODO
            # return super().mutate_return(node)
            # return loma_ir.Return(\
            # self.mutate_expr(node.val), lineno = node.lineno)

            # print("[mutate_return 1]", node)

            # * handle case: return x[i]
            # !feeling wrong with this..
            # print(f">>>> {node.val}")

            if isinstance(node.val, loma_ir.ArrayAccess) or isinstance(node.val, loma_ir.StructAccess) :
                #! trigger mutate_array_access
                new_val = self.mutate_expr(node.val)

                # print(f"[mutate_return] {new_val}")

                ret_expr = [
                    loma_ir.StructAccess(
                        new_val,
                        "val",
                    ),
                    loma_ir.StructAccess(new_val, "dval"),
                ]
                # print("[mutate_return 0]", ret_expr)

                ret_expr = loma_ir.Call("make__dfloat", ret_expr)
                return loma_ir.Return(ret_expr, lineno=node.lineno)


            ret_expr = self.mutate_expr(node.val)
            if isinstance(node.val.t, loma_ir.Int):
                ret_expr = ret_expr[0]
            elif isinstance(node.val.t, loma_ir.Struct):
                ret_expr = ret_expr[0]
            else:
                ret_expr = loma_ir.Call("make__dfloat", ret_expr)
            return loma_ir.Return(ret_expr, lineno=node.lineno)

        def mutate_declare(self, node):
            # HW1: TODO
            # return super().mutate_declare(node)
            if node.val is not None:
                val_expr = self.mutate_expr(node.val)
                if isinstance(node.val.t, loma_ir.Int):
                    val_expr = val_expr[0]
                elif isinstance(node.val.t, loma_ir.Struct):
                    val_expr = val_expr[0]
                else:
                    val_expr = loma_ir.Call("make__dfloat", val_expr)

            return loma_ir.Declare(
                node.target,
                autodiff.type_to_diff_type(diff_structs, node.t),
                val_expr if node.val is not None else None,
                lineno=node.lineno,
            )

        def mutate_assign(self, node):
            # HW1: TODO
            # return super().mutate_assign(node)

            # change var /array type to dfloat
            target_type = autodiff.type_to_diff_type(diff_structs, node.target.t)
            if isinstance(node.target, loma_ir.Var):
                target_expr = loma_ir.Var(
                    node.target.id, node.target.lineno, target_type
                )
            elif isinstance(node.target, loma_ir.ArrayAccess):
                new_inx = self.mutate_expr(node.target.index)[0]
                target_expr = loma_ir.ArrayAccess(
                    node.target.array,
                    new_inx,
                    node.target.lineno,
                    target_type,
                )
            elif isinstance(node.target, loma_ir.StructAccess):
                # print(f"mutate_assign 0 {node.target}")
                target_expr = self.mutate_expr(node.target)
                

            else:
                assert (
                        False
                    ), f"forward mode mutate_assign error: unhandled node.target {node.target}"


            assign_expr = self.mutate_expr(node.val)
            if isinstance(node.target.t, loma_ir.Int):
                assign_expr = assign_expr[0]
            elif isinstance(node.target.t, loma_ir.Struct):
                assign_expr = assign_expr[0]
            elif isinstance(assign_expr, loma_ir.StructAccess):
                struct_tup = [
                    loma_ir.StructAccess(
                        assign_expr,
                        "val",
                    ),
                    loma_ir.StructAccess(assign_expr, "dval"),
                ]
                assign_expr = loma_ir.Call("make__dfloat", struct_tup)
            else:
                assign_expr = loma_ir.Call("make__dfloat", assign_expr)

            return loma_ir.Assign(
                target_expr,
                assign_expr,
                lineno=node.lineno,
            )

        def mutate_ifelse(self, node):
            # HW3: TODO
            return super().mutate_ifelse(node)

        def mutate_while(self, node):
            # HW3: TODO
            return super().mutate_while(node)

        def mutate_const_float(self, node):
            # HW1: TODO
            # return super().mutate_const_float(node)
            # return loma_ir.Call('make__dfloat',
            # [node, loma_ir.ConstFloat(0.0)])
            return node, loma_ir.ConstFloat(0.0)

        def mutate_const_int(self, node):
            # HW1: TODO
            # * just keep returning a tuple in forward mutate_expr
            # return super().mutate_const_int(node)
            return node, loma_ir.ConstInt(0)

        def mutate_var(self, node):
            # HW1: TODO
            # return super().mutate_var(node)

            # return only one var is causing too much troubles..
            # returning a zero if it's int or struct
            if isinstance(node.t, loma_ir.Int):
                return node, loma_ir.ConstInt(0)
            if isinstance(node.t, loma_ir.Struct):
                return node, loma_ir.ConstInt(0)

            return loma_ir.StructAccess(node, "val"), loma_ir.StructAccess(node, "dval")

        def mutate_array_access(self, node):
            # HW1: TODO
            # return super().mutate_array_access(node)
            # ???? tricky...
            arr_t = autodiff.type_to_diff_type(diff_structs, node.array.t)
            idx_t = autodiff.type_to_diff_type(diff_structs, node.index.t)

            # print("debug", node)
            # print("debug", node.array)
            # print("debug 0", self.mutate_expr(node.array))
            if isinstance(node.array, loma_ir.Var):
                new_arr = loma_ir.Var(id=node.array.id, lineno=node.array.lineno, t=arr_t)
            else:
                new_arr = self.mutate_expr(node.array)

            # print(f"mutate_array_access0] {self.mutate_expr(node.index)}")
            # *index has two types: int or expr
            # for int, just use it as index
            # for expr, using mutate_expr
            new_idx = self.mutate_expr(node.index)
            if not isinstance(new_idx, loma_ir.expr): # index is Int
                new_idx = new_idx[0]
            print(f"mutate_array_access] {new_idx} vs {node.index}")

            # self.mutate_expr(new_arr)[0] is meaningless because arr.dval don't exist
            # only the element inside array has dval, e.g. arr[0].val
            val_arr_access = loma_ir.ArrayAccess(
                new_arr,
                new_idx,
                lineno=node.lineno,
                t=node.t,
            )

            return val_arr_access

        def mutate_struct_access(self, node):
            # HW1: TODO
            # return super().mutate_struct_access(node)

            print(f"[mutate_struct_access] 0: {node}")
            print(f"[mutate_struct_access] 1: {node.struct}")

            # construct a new struct
            old_struct = node.struct
            
            if not isinstance(old_struct, loma_ir.Var): # StructAccess or ArrayAccess
                new_struct = self.mutate_expr(old_struct)
            else: # struct = Var
                old_struct_type = old_struct.t 
                new_struct_type = loma_ir.Struct(
                    id= old_struct_type.id,
                    members=[loma_ir.MemberDef(
                        member.id, autodiff.type_to_diff_type(diff_structs, member.t)
                    )
                    for member in old_struct_type.members],
                    lineno=old_struct_type.lineno)

                new_struct = loma_ir.Var(
                    id=old_struct.id, t=new_struct_type, lineno=old_struct.lineno
                )

            # print(f"[mutate_struct_access] 2: {new_struct}")
            # print(f"debug: {node.struct}")

            return loma_ir.StructAccess(
                new_struct,
                node.member_id,
                lineno=node.lineno,
                t=node.t,
            )

        def binary_op_helper(self, node):
            """
            helper function, return left_val, left_dval, right_val, right_dval
            """
            if isinstance(node.left, loma_ir.ArrayAccess):
                new_arr = self.mutate_expr(node.left)
                left_val = loma_ir.StructAccess(
                    new_arr,
                    "val",
                )
                left_dval = loma_ir.StructAccess(
                    new_arr,
                    "dval",
                )
            elif isinstance(node.left, loma_ir.StructAccess):
                new_struct = self.mutate_expr(node.left)
                left_val = loma_ir.StructAccess(
                    new_struct,
                    "val",
                )
                left_dval = loma_ir.StructAccess(
                    new_struct,
                    "dval",
                )
            else:
                left_expr = self.mutate_expr(node.left)
                left_val = left_expr[0]
                left_dval = left_expr[1]

            if isinstance(node.right, loma_ir.ArrayAccess):
                new_arr = self.mutate_expr(node.right)
                right_val = loma_ir.StructAccess(
                    new_arr,
                    "val",
                )
                right_dval = loma_ir.StructAccess(
                    new_arr,
                    "dval",
                )
            elif isinstance(node.right, loma_ir.StructAccess):
                #todo: combine struc
                new_struct = self.mutate_expr(node.right)
                right_val = loma_ir.StructAccess(
                    new_struct,
                    "val",
                )
                right_dval = loma_ir.StructAccess(
                    new_struct,
                    "dval",
                )
            else:
                right_expr = self.mutate_expr(node.right)
                right_val = right_expr[0]
                right_dval = right_expr[1]
            return left_val, left_dval, right_val, right_dval

        def mutate_add(self, node):
            # HW1: TODO
            # return super().mutate_add(node)
            left_val, left_dval, right_val, right_dval =  self.binary_op_helper(node)

            val = loma_ir.BinaryOp(
                loma_ir.Add(), left_val, right_val, lineno=node.lineno, t=node.t
            )

            dval = loma_ir.BinaryOp(
                loma_ir.Add(), left_dval, right_dval, lineno=node.lineno, t=node.t
            )

            return val, dval

        def mutate_sub(self, node):
            # HW1: TODO
            # return super().mutate_sub(node)

            left_val, left_dval, right_val, right_dval =  self.binary_op_helper(node)

            val = loma_ir.BinaryOp(
                loma_ir.Sub(), left_val, right_val, lineno=node.lineno, t=node.t
            )

            dval = loma_ir.BinaryOp(
                loma_ir.Sub(), left_dval, right_dval, lineno=node.lineno, t=node.t
            )

            return val, dval

        def mutate_mul(self, node):
            # HW1: TODO
            # return super().mutate_mul(node)
            left_val, left_dval, right_val, right_dval =  self.binary_op_helper(node)

            val = loma_ir.BinaryOp(
                loma_ir.Mul(), left_val, right_val, lineno=node.lineno, t=node.t
            )

            add_op1 = loma_ir.BinaryOp(
                loma_ir.Mul(), left_val, right_dval, lineno=node.lineno, t=node.t
            )

            add_op2 = loma_ir.BinaryOp(
                loma_ir.Mul(), right_val, left_dval, lineno=node.lineno, t=node.t
            )

            dval = loma_ir.BinaryOp(
                loma_ir.Add(), add_op1, add_op2, lineno=node.lineno, t=node.t
            )

            return val, dval

        def mutate_div(self, node):
            # HW1: TODO
            # return super().mutate_div(node)
            
            left_val, left_dval, right_val, right_dval =  self.binary_op_helper(node)

            val = loma_ir.BinaryOp(
                loma_ir.Div(), left_val, right_val, lineno=node.lineno, t=node.t
            )

            sqr_y = loma_ir.BinaryOp(
                loma_ir.Mul(),
                right_val,
                right_val,
                lineno=node.lineno,
                t=node.t,
            )
            sub_op1 = loma_ir.BinaryOp(
                loma_ir.Div(),
                loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    left_val,
                    right_dval,
                    lineno=node.lineno,
                    t=node.t,
                ),
                sqr_y,
                lineno=node.lineno,
                t=node.t,
            )

            sub_op2 = loma_ir.BinaryOp(
                loma_ir.Div(),
                loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    right_val,
                    left_dval,
                    lineno=node.lineno,
                    t=node.t,
                ),
                sqr_y,
                lineno=node.lineno,
                t=node.t,
            )

            #! notice0: divided by the y^2 seperately to enforce the correct order (do`sub` first then `div`)
            #! notice1: the handout miss the minus sign, dval = - (x dy - y dx)/y^2
            dval = loma_ir.BinaryOp(
                loma_ir.Sub(), sub_op2, sub_op1, lineno=node.lineno, t=node.t
            )

            return val, dval

        def mutate_call(self, node):
            # HW1: TODO
            # return super().mutate_call(node)

            # * special case for input const_int call
            # * Notice that the mutate_expr(int) is still itself
            # * which is a int, not a tuple (don't do subscription)
            if node.id == "int2float":
                val = loma_ir.Call(node.id, node.args, lineno=node.lineno, t=node.t)
                dval = loma_ir.ConstInt(0)
                return val, dval

            # * get the args for primal function call.
            arg_vals = []
            arg_dvals = []

            # todo
            for arg in node.args:
                if isinstance(arg, loma_ir.ArrayAccess):
                    new_arg = self.mutate_expr(arg)

                    arg_vals.append(
                        loma_ir.StructAccess(
                            new_arg,
                            "val",
                        )
                    )
                    arg_dvals.append(
                        loma_ir.StructAccess(
                            new_arg,
                            "dval",
                        )
                    )
                elif isinstance(arg, loma_ir.StructAccess):
                    new_arg = self.mutate_expr(arg)

                    arg_vals.append(
                        loma_ir.StructAccess(
                            new_arg,
                            "val",
                        )
                    )
                    arg_dvals.append(
                        loma_ir.StructAccess(
                            new_arg,
                            "dval",
                        )
                    )
                else:
                    arg_vals.append(self.mutate_expr(arg)[0])
                    arg_dvals.append(self.mutate_expr(arg)[1])

            val = loma_ir.Call(node.id, arg_vals, lineno=node.lineno, t=node.t)

            dval = None
            match node.id:
                case "sin":
                    mul_op1 = loma_ir.Call(
                        "cos",
                        # [self.mutate_expr(node.args[0])[0]],
                        arg_vals,
                        lineno=node.lineno,
                        t=node.t,
                    )
                    # mul_op2 = self.mutate_expr(node.args[0])[1]
                    mul_op2 = arg_dvals[0]
                    dval = loma_ir.BinaryOp(
                        loma_ir.Mul(), mul_op1, mul_op2, lineno=node.lineno, t=node.t
                    )
                case "cos":
                    mul_op1 = loma_ir.Call(
                        "sin",
                        arg_vals,
                        lineno=node.lineno,
                        t=node.t,
                    )
                    mul_op2 = arg_dvals[0]
                    dval = loma_ir.BinaryOp(
                        loma_ir.Sub(),
                        loma_ir.ConstFloat(0.0),
                        loma_ir.BinaryOp(
                            loma_ir.Mul(),
                            mul_op1,
                            mul_op2,
                            lineno=node.lineno,
                            t=node.t,
                        ),
                        lineno=node.lineno,
                        t=node.t,
                    )
                case "sqrt":
                    dval = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        loma_ir.ConstFloat(0.5),
                        loma_ir.BinaryOp(
                            loma_ir.Div(),
                            # loma_ir.StructAccess(node.args[0], "dval"),
                            arg_dvals[0],
                            val,
                            lineno=node.lineno,
                            t=node.t,
                        ),
                        lineno=node.lineno,
                        t=node.t,
                    )
                case "pow":
                    x = arg_vals[0]
                    dx = arg_dvals[0]

                    y = arg_vals[1]
                    dy = arg_dvals[1]

                    add_op1 = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        dx,
                        loma_ir.BinaryOp(
                            loma_ir.Mul(),
                            y,
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
                            lineno=node.lineno,
                            t=node.t,
                        ),
                        lineno=node.lineno,
                        t=node.t,
                    )

                    add_op2 = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        dy,
                        loma_ir.BinaryOp(
                            loma_ir.Mul(),
                            val,
                            loma_ir.Call("log", [x], lineno=node.lineno, t=node.t),
                            lineno=node.lineno,
                            t=node.t,
                        ),
                        lineno=node.lineno,
                        t=node.t,
                    )

                    dval = loma_ir.BinaryOp(
                        loma_ir.Add(), add_op1, add_op2, lineno=node.lineno, t=node.t
                    )
                case "exp":
                    dval = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        val,
                        arg_dvals[0],
                        lineno=node.lineno,
                        t=node.t,
                    )
                case "log":
                    # print("debug:", node.args[0])
                    # print("prev:", loma_ir.StructAccess(node.args[0], "dval"))
                    # print("after:", self.mutate_expr(node.args[0])[1])
                    dval = loma_ir.BinaryOp(
                        loma_ir.Div(),
                        # loma_ir.StructAccess(node.args[0], "dval"),
                        # loma_ir.StructAccess(node.args[0], "val"),
                        arg_dvals[0],
                        arg_vals[0],
                        lineno=node.lineno,
                        t=node.t,
                    )
                # case "int2float":
                #     dval = loma_ir.ConstInt(0)
                case "float2int":
                    dval = loma_ir.ConstInt(0)
                case _:
                    assert (
                        False
                    ), f"forward mode mutate_call error: unhandled call id {node.id}"

            return val, dval

    return FwdDiffMutator().mutate_function_def(func)
