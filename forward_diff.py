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
            # print("[mutate_return] arg: ", self.mutate_expr(node.val))
            ret_expr = self.mutate_expr(node.val)
            if isinstance(node.val.t, loma_ir.Int):
                ret_expr = ret_expr[0]
            else:
                ret_expr = loma_ir.Call("make__dfloat", ret_expr)
            return loma_ir.Return(ret_expr, lineno=node.lineno)

        def mutate_declare(self, node):
            # HW1: TODO
            # return super().mutate_declare(node)
            val_expr = node.val
            if node.val is not None:
                if not isinstance(val_expr.t, loma_ir.Int):
                    val_expr = loma_ir.Call("make__dfloat", self.mutate_expr(node.val))

            return loma_ir.Declare(
                node.target,
                autodiff.type_to_diff_type(diff_structs, node.t),
                val_expr if node.val is not None else None,
                lineno=node.lineno,
            )

        def mutate_assign(self, node):
            # HW1: TODO
            # return super().mutate_assign(node)
            target_type = autodiff.type_to_diff_type(diff_structs, node.target.t)
            target_expr = loma_ir.Var(node.target.id, node.target.lineno, target_type)
            
            # todo: test assign Int
            assign_expr = node.val
            if not isinstance(node.target.t, loma_ir.Int):
                assign_expr = loma_ir.Call("make__dfloat", self.mutate_expr(node.val))

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
            return super().mutate_const_int(node)

        def mutate_var(self, node):
            # HW1: TODO
            # return super().mutate_var(node)
            # print("[mutate_var]")
            return loma_ir.StructAccess(node, "val"), loma_ir.StructAccess(node, "dval")

        def mutate_array_access(self, node):
            # HW1: TODO
            return super().mutate_array_access(node)

        def mutate_struct_access(self, node):
            # HW1: TODO
            return super().mutate_struct_access(node)

        def mutate_add(self, node):
            # HW1: TODO
            # return super().mutate_add(node)

            left_expr = self.mutate_expr(node.left)
            right_expr = self.mutate_expr(node.right)

            val = loma_ir.BinaryOp(
                loma_ir.Add(), left_expr[0], right_expr[0], lineno=node.lineno, t=node.t
            )

            dval = loma_ir.BinaryOp(
                loma_ir.Add(), left_expr[1], right_expr[1], lineno=node.lineno, t=node.t
            )

            return val, dval

        def mutate_sub(self, node):
            # HW1: TODO
            # return super().mutate_sub(node)

            left_expr = self.mutate_expr(node.left)
            right_expr = self.mutate_expr(node.right)

            val = loma_ir.BinaryOp(
                loma_ir.Sub(), left_expr[0], right_expr[0], lineno=node.lineno, t=node.t
            )

            dval = loma_ir.BinaryOp(
                loma_ir.Sub(), left_expr[1], right_expr[1], lineno=node.lineno, t=node.t
            )

            return val, dval

        def mutate_mul(self, node):
            # HW1: TODO
            # return super().mutate_mul(node)
            left_expr = self.mutate_expr(node.left)
            right_expr = self.mutate_expr(node.right)

            val = loma_ir.BinaryOp(
                loma_ir.Mul(), left_expr[0], right_expr[0], lineno=node.lineno, t=node.t
            )

            add_op1 = loma_ir.BinaryOp(
                loma_ir.Mul(), left_expr[0], right_expr[1], lineno=node.lineno, t=node.t
            )

            add_op2 = loma_ir.BinaryOp(
                loma_ir.Mul(), right_expr[0], left_expr[1], lineno=node.lineno, t=node.t
            )

            dval = loma_ir.BinaryOp(
                loma_ir.Add(), add_op1, add_op2, lineno=node.lineno, t=node.t
            )

            return val, dval

        def mutate_div(self, node):
            # HW1: TODO
            # return super().mutate_div(node)
            left_expr = self.mutate_expr(node.left)
            right_expr = self.mutate_expr(node.right)

            val = loma_ir.BinaryOp(
                loma_ir.Div(), left_expr[0], right_expr[0], lineno=node.lineno, t=node.t
            )

            sqr_y = loma_ir.BinaryOp(
                loma_ir.Mul(),
                right_expr[0],
                right_expr[0],
                lineno=node.lineno,
                t=node.t,
            )
            sub_op1 = loma_ir.BinaryOp(
                loma_ir.Div(),
                loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    left_expr[0],
                    right_expr[1],
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
                    right_expr[0],
                    left_expr[1],
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

            # * get the args for primal function call. 
            if node.id == "int2float" or node.id == "int2float":
                # * Notice that the mutate_expr(const_int) is still itself (not a tuple)
                val = loma_ir.Call(node.id, node.args, lineno=node.lineno, t=node.t)
            else:
                # * Notice that the mutate_var() return a tuple now
                args = [self.mutate_expr(arg)[0] for arg in node.args]
                val = loma_ir.Call(node.id, args, lineno=node.lineno, t=node.t)

            dval = None
            match node.id:
                case "sin":
                    mul_op1 = loma_ir.Call(
                        "cos",
                        [self.mutate_expr(node.args[0])[0]],
                        lineno=node.lineno,
                        t=node.t,
                    )
                    mul_op2 = self.mutate_expr(node.args[0])[1]
                    dval = loma_ir.BinaryOp(
                        loma_ir.Mul(), mul_op1, mul_op2, lineno=node.lineno, t=node.t
                    )
                case "cos":
                    mul_op1 = loma_ir.Call(
                        "sin",
                        [self.mutate_expr(node.args[0])[0]],
                        lineno=node.lineno,
                        t=node.t,
                    )
                    mul_op2 = self.mutate_expr(node.args[0])[1]
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
                            self.mutate_expr(node.args[0])[1],
                            val,
                            lineno=node.lineno,
                            t=node.t,
                        ),
                        lineno=node.lineno,
                        t=node.t,
                    )
                case "pow":
                    x = self.mutate_expr(node.args[0])[0]
                    dx = self.mutate_expr(node.args[0])[1]

                    y = self.mutate_expr(node.args[1])[0]
                    dy = self.mutate_expr(node.args[1])[1]

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
                        self.mutate_expr(node.args[0])[1],
                        lineno=node.lineno,
                        t=node.t,
                    )
                case "log":
                    dval = loma_ir.BinaryOp(
                        loma_ir.Div(),
                        # loma_ir.StructAccess(node.args[0], "dval"),
                        # loma_ir.StructAccess(node.args[0], "val"),
                        self.mutate_expr(node.args[0])[1],
                        self.mutate_expr(node.args[0])[0],
                        lineno=node.lineno,
                        t=node.t,
                    )
                case "int2float":
                    dval = loma_ir.ConstInt(0)
                case "float2int":
                    dval = loma_ir.ConstInt(0)
                case _:
                    assert (
                        False
                    ), f"forward mode mutate_call error: unhandled call id {node.id}"

            return val, dval

    return FwdDiffMutator().mutate_function_def(func)
