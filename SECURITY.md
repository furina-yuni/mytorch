# Security policy

## Supported versions

Security fixes are provided for the latest released 0.x version only.

## Reporting a vulnerability

Do not open a public issue containing exploit details, credentials, model
checkpoints, or private data. After the repository is published, enable GitHub
private vulnerability reporting under **Settings → Security → Code security**
and submit reports there. Until that channel is configured, distribute this
project only to trusted testers.

Include the affected version, operating system, CUDA/CuPy versions, a minimal
reproduction, and the expected impact. Never attach proprietary training data
or secrets.

MyTorch checkpoint files use NPZ arrays plus validated metadata and do not
deserialize arbitrary Python pickle objects. Checkpoints are still untrusted
input: validate their origin and keep size limits appropriate to your system.
