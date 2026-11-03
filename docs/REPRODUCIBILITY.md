# Reproducibility and evidence

V3 records MuJoCo Python/engine versions, friction, mass, timestep, seed, task, scene, split, full initial state bytes, source revision, actions, state/velocity/contact, RGB SHA-256 and JSON SHA-256. Validation checks hashes, exact branch starts and identical initial physics states across frozen splits. Generation and training refuse existing output paths.

Environment: Windows, Python 3.12, torch 2.15.0+cpu, MuJoCo 3.17.0; GeForce GTX 1650, 4096 MiB, CUDA unavailable in installed wheel. Initial disk free: about 296 GB. Linux is expected but untested. Pre-v2 pilot scenes repeated physical initial states across split labels; local outputs are retained for audit and must not be comparative evidence.
