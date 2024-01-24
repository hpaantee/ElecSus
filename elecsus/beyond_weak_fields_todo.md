Open issues:
- [transit-time] why is there a factor between simulation and expectation
- [transit-time] what is the general formula
- [transit-time] is the effect uniform for both g/e and g/g transitions
- [collisions] dephasing or decay
- [collisions] how different is it for g/e and g/g
- [collisions] mismatch between simulation and experiment
- [collisions/transit-time] do collisions increase the interaction time with the beam? Can this explain the factor ~4?
- [propagation] curves do not match so far
...
- code cleanup
...
- Rename: laserWaist ist actually laserDiameter
- How to calculate isotope shifts? Currently taken from elecsus/libs/AtomConstants.py (different for D1/D2)
- If we remove 'Bfield':0.,'Btheta':0., 'Bphi':0.,'GammaBuf':0.,'shift':0 from p_dict (already non-bwf),
  we get "ValueError: array must not contain infs or Nans"
- Why does fit_data use strange E-field definition (
    params['E_x'].value = E_in[0]
	params['E_y'].value = E_in[1][0]
	params['E_phase'].value = E_in[1][1])








KEYWORDS:
- velocity changing collisions