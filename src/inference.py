"""SageMaker SKLearnModel entry-point shim.

SageMaker's `sagemaker_containers._modules.import_module` calls
`importlib.import_module(name)` where `name` is the entry_point with `.py`
stripped — slashes are not converted to dots. So `entry_point` must be a bare
filename (relative to `source_dir`), which is why this shim lives at
`src/inference.py` rather than `src/readmit/models/inference.py`.

The real handlers live under `readmit.models.inference`; this file just
re-exports them so the four-handler contract (model_fn, input_fn, predict_fn,
output_fn) is satisfied at the location SageMaker expects.
"""
from readmit.models.inference import (  # noqa: F401
    input_fn,
    model_fn,
    output_fn,
    predict_fn,
)
