import pyEXP

discbasisconfig = """
id: cylinder
parameters:
  acyl: 1.0
  hcyl: 0.1
  mmax: 10
  ncylorder: 32
  nmax: 48
  ncylnx: 256
  ncylny: 128
  rnum: 200
  pnum: 1
  tnum: 80
  logr: true
  density: true
  eof_file: .eof.cache.file.h5
  rcylmin : 0.001
  rcylmax : 20.0
  """
discbasis = pyEXP.basis.Basis.factory(discbasisconfig)
