SKIP: INVALID


RWByteAddressBuffer prevent_dce : register(u0);
int4 subgroupAnd_97655b() {
  int4 arg_0 = (int(1)).xxxx;
  int4 res = asint(WaveActiveBitAnd(asuint(arg_0)));
  return res;
}

void fragment_main() {
  prevent_dce.Store4(0u, asuint(subgroupAnd_97655b()));
}

[numthreads(1, 1, 1)]
void compute_main() {
  prevent_dce.Store4(0u, asuint(subgroupAnd_97655b()));
}

FXC validation failure:
<scrubbed_path>(5,20-50): error X3004: undeclared identifier 'WaveActiveBitAnd'


tint executable returned error: exit status 1
