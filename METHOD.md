# Fork-and-flush rounds

Late in the challenge I tried the fork-and-flush method from *Fork-and-Flush: Escaping Idea Basins in Autoresearch Agents* (arXiv 2610.07447). The idea: separate runs of the same agent tend to get stuck circling their own set of ideas. So every so often you fork the agent into parallel runs that keep the workspace but start with a fresh chat, then carry on from the best one.

**Round 1, decode layer.** Three fresh agents worked from the same tuned parent. They had the same brief and a small, fixed budget of judge checks each. Two came back with real savings. Merged, the result read +0.26% and +0.37% in two paired checks. The board row moved from 2.268× to 2.297×, which is the decode-layer file here.

**Later rounds.** A second round on the decode layer, and one round each on MoE, prefill, projections and attention, added nothing to the board scores. The MoE round found a real +0.36%, but that change only copied code the layer already had, and its one scored run came in below the existing row.

One good round is weak evidence for the method. The first round on an already tuned file probably picked up the easiest remaining gains.
