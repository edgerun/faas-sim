from sim.stats import ParameterizedDistribution as PDist

execution_time_distributions = {
    ('cloud', 'alexrashed/ml-wf-1-pre:0.37'): (0.584, 1.1420000000000001, PDist.lognorm(((0.31449780857108944,), 0.36909628354997315, 0.41583220283981315))),
    ('tegra', 'alexrashed/ml-wf-1-pre:0.37'): (0.434, 0.491, PDist.lognorm(((0.4040891723912467,), 0.42388853817361616, 0.026861281861394234))),
    ('pi', 'alexrashed/ml-wf-1-pre:0.37'): (23.89, 108.743, PDist.lognorm(((0.20506023311489027,), -20.283277542855338, 73.25412435048207))),
    ('nuc', 'alexrashed/ml-wf-1-pre:0.37'): (0.10099999999999999, 0.105, PDist.lognorm(((23.561803013427763,), 0.10099999999999998, 0.00012383535141814733))),
    ('cloud', 'alexrashed/ml-wf-2-train:0.37'): (156.256, 166.702, PDist.lognorm(((0.0955189827829152,), 140.38073254297746, 20.5799433435967))),
    ('tegra', 'alexrashed/ml-wf-2-train:0.37'): (17.641, 20.333, PDist.lognorm(((0.07553515034079655,), 12.501696816822736, 6.339535271582099))),
    ('nuc', 'alexrashed/ml-wf-2-train:0.37'): (31.326999999999998, 42.355, PDist.lognorm(((0.08117635531817702,), 4.01869140845729, 32.19443389130219))),
    ('cloud', 'alexrashed/ml-wf-3-serve:0.37'): (0.542, 11.505, PDist.lognorm(((2.148861851180003,), 0.5402853007866957, 1.422846600303037))),
    ('tegra', 'alexrashed/ml-wf-3-serve:0.37'): (0.48, 10.882, PDist.lognorm(((4.521620371690464,), 0.4799999929591894, 5.3308588402679185))),
    ('pi', 'alexrashed/ml-wf-3-serve:0.37'): (1.203, 14.579, PDist.lognorm(((0.0011447901398451941,), -2722.59408910652, 2733.773971903728))),
    ('nuc', 'alexrashed/ml-wf-3-serve:0.37'): (0.14400000000000002, 10.261, PDist.lognorm(((3.0008797752241914,), 0.14399171198249225, 0.0833885862659985))),
}

startup_time_distributions = {
    ('cloud', 'alexrashed/ml-wf-1-pre:0.37', False, 1250000.0): (
        522.1168, 626.0352, PDist.lognorm(((0.8128,), 520.7708, 8.9967))),
    ('cloud', 'alexrashed/ml-wf-1-pre:0.37', False, 12500000.0):
        (112.7926, 145.9801, PDist.lognorm(((0.3541,), 106.444, 13.9728))),
    ('cloud', 'alexrashed/ml-wf-1-pre:0.37', False, 25000000.0):
        (95.2460, 111.4869, PDist.lognorm(((0.4819,), 92.8929, 7.6889))),
    ('cloud', 'alexrashed/ml-wf-1-pre:0.37', False, 125000000.0):
        (93.0299, 165.6075, PDist.lognorm(((0.5815,), 90.8732, 11.2789))),
    ('cloud', 'alexrashed/ml-wf-1-pre:0.37', True, 1250000.0):
        (4.5509, 7.1673, PDist.lognorm(((0.6474,), 4.2865, 1.0156))),
    ('cloud', 'alexrashed/ml-wf-1-pre:0.37', True, 12500000.0):
        (4.4578, 7.0044, PDist.lognorm(((0.4078,), 3.9476, 1.366))),
    ('cloud', 'alexrashed/ml-wf-1-pre:0.37', True, 25000000.0):
        (4.5443, 7.1637, PDist.lognorm(((0.5186,), 4.2054, 1.2562))),
    ('cloud', 'alexrashed/ml-wf-1-pre:0.37', True, 125000000.0):
        (4.7599, 8.0136, PDist.lognorm(((0.0072,), -96.3933, 102.7861))),
    ('cloud', 'alexrashed/ml-wf-2-train:0.37', False, 1250000.0):
        (537.5693, 863.9742, PDist.lognorm(((0.8986,), 536.4493, 9.4342))),
    ('cloud', 'alexrashed/ml-wf-2-train:0.37', False, 12500000.0):
        (116.2464, 161.2726, PDist.lognorm(((0.5562,), 113.9494, 9.1997))),
    ('cloud', 'alexrashed/ml-wf-2-train:0.37', False, 25000000.0):
        (97.3947, 171.4834, PDist.lognorm(((0.7065,), 96.1597, 7.4872))),
    ('cloud', 'alexrashed/ml-wf-2-train:0.37', False, 125000000.0):
        (92.9383, 188.8597, PDist.lognorm(((0.601,), 90.4327, 12.329))),
    ('cloud', 'alexrashed/ml-wf-2-train:0.37', True, 1250000.0):
        (4.5556, 7.5936, PDist.lognorm(((0.6259,), 4.357, 0.8369))),
    ('cloud', 'alexrashed/ml-wf-2-train:0.37', True, 12500000.0):
        (4.6167, 6.9819, PDist.lognorm(((0.4952,), 4.2045, 1.1753))),
    ('cloud', 'alexrashed/ml-wf-2-train:0.37', True, 25000000.0):
        (4.5653, 7.5886, PDist.lognorm(((0.7536,), 4.4536, 0.7464))),
    ('cloud', 'alexrashed/ml-wf-2-train:0.37', True, 125000000.0):
        (4.6123, 7.2525, PDist.lognorm(((0.6097,), 4.4895, 0.685))),
    ('cloud', 'alexrashed/ml-wf-3-serve:0.37', False, 1250000.0):
        (573.4476, 2428.3331, PDist.lognorm(((1.0537,), 572.8704, 10.116))),
    ('cloud', 'alexrashed/ml-wf-3-serve:0.37', False, 12500000.0):
        (122.4523, 140.7086, PDist.lognorm(((0.1541,), 101.46, 29.4202))),
    ('cloud', 'alexrashed/ml-wf-3-serve:0.37', False, 25000000.0):
        (99.7786, 121.4573, PDist.lognorm(((0.3205,), 92.6263, 16.2183))),
    ('cloud', 'alexrashed/ml-wf-3-serve:0.37', False, 125000000.0):
        (98.2754, 3529.2718, PDist.lognorm(((1.1226,), 97.9458, 8.3177))),
    ('cloud', 'alexrashed/ml-wf-3-serve:0.37', True, 1250000.0):
        (4.6761, 9.0359, PDist.lognorm(((0.7024,), 4.4737, 1.0673))),
    ('cloud', 'alexrashed/ml-wf-3-serve:0.37', True, 12500000.0):
        (4.6657, 7.2533, PDist.lognorm(((0.4991,), 4.2268, 1.2579))),
    ('cloud', 'alexrashed/ml-wf-3-serve:0.37', True, 25000000.0):
        (4.6696, 7.9862, PDist.lognorm(((0.5383,), 4.3766, 1.1787))),
    ('cloud', 'alexrashed/ml-wf-3-serve:0.37', True, 125000000.0):
        (4.6732, 8.0709, PDist.lognorm(((0.6307,), 4.4997, 0.8694))),
    ('pi', 'alexrashed/ml-wf-1-pre:0.37', False, 1250000.0):
        (743.1166, 2678.2766, PDist.lognorm(((0.8152,), 725.1841, 208.5534))),
    ('pi', 'alexrashed/ml-wf-1-pre:0.37', False, 12500000.0):
        (464.7706, 3223.8342, PDist.lognorm(((5.9356,), 464.7706, 1.4296))),
    ('pi', 'alexrashed/ml-wf-1-pre:0.37', False, 25000000.0):
        (481.3685, 1911.9403, PDist.lognorm(((0.7853,), 454.2561, 234.8997))),
    ('pi', 'alexrashed/ml-wf-1-pre:0.37', False, 125000000.0):
        (439.7734, 1217.8021, PDist.lognorm(((0.5657,), 396.4158, 249.5016))),
    ('pi', 'alexrashed/ml-wf-1-pre:0.37', True, 1250000.0):
        (2.6007, 48.3604, PDist.lognorm(((0.8615,), 2.3423, 2.2614))),
    ('pi', 'alexrashed/ml-wf-1-pre:0.37', True, 12500000.0):
        (2.5354, 20.1571, PDist.lognorm(((0.8443,), 2.1801, 2.3138))),
    ('pi', 'alexrashed/ml-wf-1-pre:0.37', True, 25000000.0):
        (2.5326, 36.5221, PDist.lognorm(((0.9983,), 2.3122, 2.1348))),
    ('pi', 'alexrashed/ml-wf-1-pre:0.37', True, 125000000.0):
        (2.5397, 20.5341, PDist.lognorm(((0.7596,), 2.1664, 2.586))),
    ('pi', 'alexrashed/ml-wf-2-train:0.37', False, 1250000.0):
        (806.7358, 2205.1962, PDist.lognorm(((0.8707,), 782.0461, 197.4753))),
    ('pi', 'alexrashed/ml-wf-2-train:0.37', False, 12500000.0):
        (505.8370, 2221.1437, PDist.lognorm(((4.8389,), 505.837, 2.7254))),
    ('pi', 'alexrashed/ml-wf-2-train:0.37', False, 25000000.0):
        (518.3794, 2539.7835, PDist.lognorm(((4.4405,), 518.3794, 0.6502))),
    ('pi', 'alexrashed/ml-wf-2-train:0.37', False, 125000000.0):
        (509.1512, 2191.0246, PDist.lognorm(((5.1079,), 509.1512, 1.1536))),
    ('pi', 'alexrashed/ml-wf-2-train:0.37', True, 1250000.0):
        (2.6216, 19.0769, PDist.lognorm(((0.9079,), 2.4359, 1.811))),
    ('pi', 'alexrashed/ml-wf-2-train:0.37', True, 12500000.0):
        (2.4477, 35.2054, PDist.lognorm(((0.9683,), 2.2859, 1.9477))),
    ('pi', 'alexrashed/ml-wf-2-train:0.37', True, 25000000.0):
        (2.5144, 21.4415, PDist.lognorm(((0.7835,), 2.0474, 2.4785))),
    ('pi', 'alexrashed/ml-wf-2-train:0.37', True, 125000000.0):
        (2.4345, 34.6909, PDist.lognorm(((0.7921,), 2.2377, 2.6027))),
    ('pi', 'alexrashed/ml-wf-3-serve:0.37', False, 1250000.0):
        (825.8014, 2532.0387, PDist.lognorm(((5.8151,), 825.8014, 1.062))),
    ('pi', 'alexrashed/ml-wf-3-serve:0.37', False, 12500000.0):
        (554.1878, 3402.3931, PDist.lognorm(((0.9273,), 536.5606, 213.9515))),
    ('pi', 'alexrashed/ml-wf-3-serve:0.37', False, 25000000.0):
        (532.5763, 2813.9212, PDist.lognorm(((6.2533,), 532.5763, 0.4951))),
    ('pi', 'alexrashed/ml-wf-3-serve:0.37', False, 125000000.0):
        (492.3203, 2175.0496, PDist.lognorm(((4.947,), 492.3203, 0.5644))),
    ('pi', 'alexrashed/ml-wf-3-serve:0.37', True, 1250000.0):
        (2.6303, 14.8142, PDist.lognorm(((0.7978,), 2.3983, 1.7395))),
    ('pi', 'alexrashed/ml-wf-3-serve:0.37', True, 12500000.0):
        (2.5173, 26.1551, PDist.lognorm(((0.8706,), 2.2578, 2.143))),
    ('pi', 'alexrashed/ml-wf-3-serve:0.37', True, 25000000.0):
        (2.5557, 45.5601, PDist.lognorm(((0.9855,), 2.3556, 2.0786))),
    ('pi', 'alexrashed/ml-wf-3-serve:0.37', True, 125000000.0):
        (2.4607, 39.2759, PDist.lognorm(((0.9608,), 2.2635, 2.2435))),
    ('tegra', 'alexrashed/ml-wf-1-pre:0.37', False, 1250000.0):
        (502.1906, 1059.3005, PDist.lognorm(((0.8541,), 500.8819, 12.0127))),
    ('tegra', 'alexrashed/ml-wf-1-pre:0.37', False, 12500000.0):
        (83.8914, 122.3928, PDist.lognorm(((0.2454,), 63.8755, 33.0379))),
    ('tegra', 'alexrashed/ml-wf-1-pre:0.37', False, 25000000.0):
        (61.1453, 109.0388, PDist.lognorm(((0.2151,), 35.7621, 40.4832))),
    ('tegra', 'alexrashed/ml-wf-1-pre:0.37', False, 125000000.0):
        (57.6945, 131.2093, PDist.lognorm(((0.6056,), 55.9627, 8.4486))),
    ('tegra', 'alexrashed/ml-wf-1-pre:0.37', True, 1250000.0):
        (1.6602, 2.9865, PDist.lognorm(((0.003,), -100.9381, 103.3403))),
    ('tegra', 'alexrashed/ml-wf-1-pre:0.37', True, 12500000.0):
        (1.5326, 2.8402, PDist.lognorm(((0.004,), -69.5941, 71.8316))),
    ('tegra', 'alexrashed/ml-wf-1-pre:0.37', True, 25000000.0):
        (1.4962, 2.8140, PDist.lognorm(((0.0031,), -97.9706, 100.219))),
    ('tegra', 'alexrashed/ml-wf-1-pre:0.37', True, 125000000.0):
        (1.4735, 3.7977, PDist.lognorm(((0.1845,), 0.0188, 2.1533))),
    ('tegra', 'alexrashed/ml-wf-2-train:0.37', False, 1250000.0):
        (537.7253, 610.1938, PDist.lognorm(((0.3906,), 531.3148, 20.8604))),
    ('tegra', 'alexrashed/ml-wf-2-train:0.37', False, 12500000.0):
        (88.4637, 154.7214, PDist.lognorm(((0.4655,), 83.755, 17.8609))),
    ('tegra', 'alexrashed/ml-wf-2-train:0.37', False, 25000000.0):
        (63.5209, 123.4705, PDist.lognorm(((0.3411,), 52.124, 25.1285))),
    ('tegra', 'alexrashed/ml-wf-2-train:0.37', False, 125000000.0):
        (58.6469, 109.4379, PDist.lognorm(((0.4053,), 49.5123, 22.3873))),
    ('tegra', 'alexrashed/ml-wf-2-train:0.37', True, 1250000.0):
        (1.6347, 2.7660, PDist.lognorm(((0.0042,), -67.2922, 69.6105))),
    ('tegra', 'alexrashed/ml-wf-2-train:0.37', True, 12500000.0):
        (1.5756, 3.5434, PDist.lognorm(((0.0908,), -0.9835, 3.2979))),
    ('tegra', 'alexrashed/ml-wf-2-train:0.37', True, 25000000.0):
        (1.4936, 2.6103, PDist.lognorm(((0.0027,), -103.2635, 105.4745))),
    ('tegra', 'alexrashed/ml-wf-2-train:0.37', True, 125000000.0):
        (1.3829, 2.6937, PDist.lognorm(((0.0024,), -108.3046, 110.4397))),
    ('tegra', 'alexrashed/ml-wf-3-serve:0.37', False, 1250000.0):
        (538.3405, 806.6493, PDist.lognorm(((0.5266,), 534.9129, 21.1772))),
    ('tegra', 'alexrashed/ml-wf-3-serve:0.37', False, 12500000.0):
        (88.0359, 138.5978, PDist.lognorm(((0.1186,), 40.1892, 64.6507))),
    ('tegra', 'alexrashed/ml-wf-3-serve:0.37', False, 25000000.0):
        (69.6726, 104.7837, PDist.lognorm(((2.5817,), 69.6713, 1.1456))),
    ('tegra', 'alexrashed/ml-wf-3-serve:0.37', False, 125000000.0):
        (58.2217, 98.4122, PDist.lognorm(((0.0987,), -21.4363, 95.8067))),
    ('tegra', 'alexrashed/ml-wf-3-serve:0.37', True, 1250000.0):
        (1.6478, 2.9493, PDist.lognorm(((0.0045,), -61.5837, 63.9677))),
    ('tegra', 'alexrashed/ml-wf-3-serve:0.37', True, 12500000.0):
        (1.5380, 2.8221, PDist.lognorm(((0.0094,), -28.4853, 30.6508))),
    ('tegra', 'alexrashed/ml-wf-3-serve:0.37', True, 25000000.0):
        (1.4548, 2.7817, PDist.lognorm(((0.0031,), -80.4446, 82.5954))),
    ('tegra', 'alexrashed/ml-wf-3-serve:0.37', True, 125000000.0):
        (1.4098, 2.7206, PDist.lognorm(((0.0052,), -56.7611, 58.8382))),
}
