# Subpolicy experiments

The blue outfield player attacks the goal at negative x. Each checkpoint must
run on every drill with the same observation and action interface. The other
active players use scripts. Record the initial poses, velocities and script
parameters so that a candidate and its competitors see the same trials.

## How to interpret the comparison

Use fresh training starts and seed 31001 for validation. Select checkpoints
using validation only. Reserve seed 91009 and at least 200 trials per drill
for the final comparison. Run every selected checkpoint and the scripted player
on each drill. Keep the simulator configuration and trial set fixed within each
comparison. A return on one drill has different units and scale from the
return on another. Compare policies on the same starts for each drill.

Report paired differences and 95% bootstrap intervals over starts. Report
actual goals, concessions, boundary exits and possession alongside return.
For Mid Field, check progress toward goal and scoring. For Ball Duel, inspect
possession against each opponent mode and whether control leads to progress.
For Positioning, count received passes and inspect passing lanes and spacing.
For Defense, count conceded goals, recoveries and clearance distance.

A positive difference against every other specialist on its home drill
supports specialization. An interval crossing zero leaves that comparison
unresolved. A good generalist may legitimately win multiple drills. Keep
the original test set and report those results even when a specialist loses.
