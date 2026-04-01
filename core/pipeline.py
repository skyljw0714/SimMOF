
import json
from pathlib import Path
from typing import Callable, Dict, Any, List, Tuple, Optional

from langchain_core.runnables import RunnableLambda

Step = Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]]

def make_pipeline_chain(
    steps: List[Step],
    dump_step: Optional[Callable[[Dict[str, Any], str, int], None]] = None,
):
    chain = RunnableLambda(lambda ctx: ctx)
    for i, (name, fn) in enumerate(steps, 1):
        chain = chain | RunnableLambda(fn)
        if dump_step:
            chain = chain | RunnableLambda(lambda ctx, n=name, k=i: (dump_step(ctx, n, k) or ctx))
    return chain
