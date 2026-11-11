"""SQL UDF typing and bag-preserving join collation at the native API boundary."""

from __future__ import annotations


def install():
    import pandas as pd
    from nirvana.lineage.abstractions import LineageNode, NodeOutput
    from nirvana.ops.join import JoinOperation, JoinOpOutputs
    from nirvana.ops.map import MapOperation

    from experiments.system_comparison.execution.operators import SQLJoin
    from experiments.validation.runtime.native_graph import Scalar

    if getattr(LineageNode, "_comparison_sql_compat", False):
        return
    original_join = JoinOperation.execute
    original_map = MapOperation._postprocess_map_output
    original_collate = LineageNode.collate_dataframe
    original_run = LineageNode.run

    async def join(self, left_data, right_data, **kwargs):
        tool = getattr(self.tool, "func", self.tool)
        if isinstance(tool, SQLJoin):
            result = JoinOpOutputs(cost=0.0)
            result.sql_frame = tool.frames(left_data, right_data)
            return result
        if left_data.empty or right_data.empty:
            # Upstream passes an unsupported `output` argument here. An
            # inner join with either input empty has no accepted pairs.
            return JoinOpOutputs(cost=0.0)
        return await original_join(self, left_data, right_data, **kwargs)

    def mapped(self, value, columns):
        tool = getattr(self.tool, "func", self.tool)
        if isinstance(tool, Scalar):
            if not isinstance(value, dict) or set(value) != set(columns):
                raise ValueError("Deterministic SQL UDF output schema mismatch")
            return value
        return original_map(self, value, columns)

    async def collate(self, input=None, op_outputs=None):
        tool = getattr(getattr(self.operator, "tool", None), "func", None)
        if self.op_name == "map" and input.empty:
            # A zero-row map still declares its output columns. Upstream's
            # empty output dict otherwise drops them after a selective filter.
            output = input.copy()
            for column in self.operator.output_columns:
                if column not in output:
                    output[column] = pd.Series(index=output.index, dtype=object)
            return output
        if self.op_name == "map" and isinstance(tool, Scalar):
            mapped = pd.DataFrame(op_outputs.outputs, index=input.index, dtype=object)
            return pd.concat(
                [input.drop(columns=list(op_outputs.outputs), errors="ignore"), mapped],
                axis=1,
            )
        if self.op_name != "join":
            return await original_collate(self, input, op_outputs)

        if hasattr(op_outputs, "sql_frame"):
            return op_outputs.sql_frame
        if self.operator.how != "inner":
            raise ValueError("Common SQL graph requires inner joins")
        # Upstream's scalar key mapping overwrites one-to-many matches. The
        # actual native accepted pair list retains the complete SQL bag.
        pairs = op_outputs.join_pairs
        if not pairs:
            return pd.concat([input[0].iloc[:0], input[1].iloc[:0]], axis=1)
        left, right = zip(*pairs)
        return pd.concat(
            [
                input[0].loc[list(left)].reset_index(drop=True),
                input[1].loc[list(right)].reset_index(drop=True),
            ],
            axis=1,
        )

    async def run(self, input=None):
        tool = getattr(getattr(self.operator, "tool", None), "func", None)
        if self.op_name == "map" and input.empty:
            return NodeOutput(
                output=await self.collate_dataframe(input, None), cost=0.0
            )
        if self.op_name != "join" and not (
            self.op_name == "map" and isinstance(tool, Scalar)
        ):
            return await original_run(self, input)
        result = await self.execute_operation(input)
        return NodeOutput(
            output=await self.collate_dataframe(input, result), cost=result.cost
        )

    JoinOperation.execute = join
    MapOperation._postprocess_map_output = mapped
    LineageNode.collate_dataframe = collate
    LineageNode.run = run
    LineageNode._comparison_sql_compat = True
