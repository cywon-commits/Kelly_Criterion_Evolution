Code archive: "Extrinsic mortality per ruin time decides whether evolution bets less or more than Kelly"
Python 3 (numpy, scipy, matplotlib). Parameters and seeds are set inside each script.

Core
  cont.py        agent-based simulation (mean field / lattice), lognormal returns; chain helpers (lnparams, kelly)
  robust.py      discretized log-wealth chain: kernel, invasion fitness R0 for vacancy / lottery / mixed / (v,mu)
  chainT.py      chain solver for reproduction probability and lifetime
  rob2.py        ESS by pessimization (vacancy: min v*, lottery: max mu*)
  family.py      renewal formula with chain ingredients (Pi, pi, T_r); curves f*(mu) for (Q,b) families
  xtheory.py, qsd.py, xanal.py   closed-form diffusion ingredients (Eqs. 5-7) and renewal ESS
  hcap.py, p0grid.py, vstarcmp.py   validation of the factorization (Eq. 2) and the QS waiting factor (Eq. 3)
Figures
  fig1b.py (Fig. 1), figs.py (Figs. 2, 3, 6, 7 and S3), figX2.py (Fig. 4), figfam.py (Fig. 5),
  figS1.py (Fig. S1), figS2.py (Fig. S2)
Simulations behind figures
  mix.py (alpha scan), phi.py (crossing data), lotscan.py / lotscan2.py (Law II scan), ecsim.py (extinction ramps),
  hubm.py / hubp.py / hubk.py / pairapp.py / sitechain.py / graphsc.py (graphs, pair approximation),
  ntheory.py and canonical-equation scripts (Fig. 7 freezing)
