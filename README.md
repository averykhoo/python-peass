# PEASS Toolkit (Python Port)

This project was ported by Gemini 3.5 Flash from
https://gitlab.inria.fr/bass-db/peass/-/tree/22c7fc4ef670f8bb6eea9ab4abea98323006b769/v2.0.1
(also in [peass_master_22c7fc4e](references/peass_master_22c7fc4e) for the LLM to refer to)

See also:
- https://www.audiolabs-erlangen.de/resources/2019-WASPAA-SEBASS
- https://www.atiam.ircam.fr/Archives/Stages1112/FRITSCH.pdf

It most ports the decomposition, summarized into `peass_decomposition.py` which is known to run to completion, 
but all the other files are also ported into [peass_direct_python_port/](references/legacy_direct_python_port),
including a conversion from .mat weights to .npz

## TODO

- use the new npz files in the port
- 