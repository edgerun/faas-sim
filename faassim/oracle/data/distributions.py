from sim.stats import ParameterizedDistribution as PDist

execution_time_distributions = {
    ('cloud', 'alexrashed/ml-wf-1-pre:0.37', 1250000.0):
        (35.0520, 106.0075, PDist.lognorm(((0.9214,), 34.8539, 3.1212))),
    ('cloud', 'alexrashed/ml-wf-1-pre:0.37', 12500000.0):
        (26.4979, 43.2985, PDist.lognorm(((0.4945,), 24.8201, 5.2022))),
    ('cloud', 'alexrashed/ml-wf-1-pre:0.37', 25000000.0):
        (23.9430, 45.5589, PDist.lognorm(((0.3668,), 21.6322, 7.2081))),
    ('cloud', 'alexrashed/ml-wf-1-pre:0.37', 125000000.0):
        (23.7877, 40.2138, PDist.lognorm(((0.2339,), 18.8262, 10.3541))),
    ('cloud', 'alexrashed/ml-wf-2-train:0.37', 1250000.0):
        (161.7466, 234.8019, PDist.lognorm(((0.5041,), 159.3251, 10.9032))),
    ('cloud', 'alexrashed/ml-wf-2-train:0.37', 12500000.0):
        (161.5305, 272.8669, PDist.lognorm(((0.6834,), 160.0346, 9.4709))),
    ('cloud', 'alexrashed/ml-wf-2-train:0.37', 25000000.0):
        (160.1912, 283.0286, PDist.lognorm(((0.602,), 158.5165, 10.5019))),
    ('cloud', 'alexrashed/ml-wf-2-train:0.37', 125000000.0):
        (161.0026, 290.7546, PDist.lognorm(((0.6283,), 159.602, 9.4187))),
    ('cloud', 'alexrashed/ml-wf-3-serve:0.37', 1250000.0):
        (0.1709, 1.1557, PDist.lognorm(((0.6783,), 0.1609, 0.075))),
    ('cloud', 'alexrashed/ml-wf-3-serve:0.37', 12500000.0):
        (0.1199, 0.9825, PDist.lognorm(((0.9192,), 0.1134, 0.0545))),
    ('cloud', 'alexrashed/ml-wf-3-serve:0.37', 25000000.0):
        (0.1045, 0.9444, PDist.lognorm(((1.3022,), 0.1028, 0.0393))),
    ('cloud', 'alexrashed/ml-wf-3-serve:0.37', 125000000.0):
        (0.0923, 0.3538, PDist.lognorm(((0.8983,), 0.0854, 0.0523))),
    ('pi', 'alexrashed/ml-wf-1-pre:0.37', 1250000.0):
        (80.5017, 475.0705, PDist.lognorm(((3.2762,), 80.5016, 0.9519))),
    ('pi', 'alexrashed/ml-wf-1-pre:0.37', 12500000.0):
        (62.5497, 163.0830, PDist.lognorm(((0.1993,), 8.6942, 92.8347))),
    ('pi', 'alexrashed/ml-wf-1-pre:0.37', 25000000.0):
        (54.6678, 396.2044, PDist.lognorm(((0.2957,), 25.6006, 70.0605))),
    ('pi', 'alexrashed/ml-wf-1-pre:0.37', 125000000.0):
        (60.6789, 775.5735, PDist.lognorm(((0.7683,), 57.7211, 44.9378))),
    ('pi', 'alexrashed/ml-wf-3-serve:0.37', 1250000.0):
        (0.3630, 3.9866, PDist.lognorm(((0.4865,), 0.352, 0.0865))),
    ('pi', 'alexrashed/ml-wf-3-serve:0.37', 12500000.0):
        (0.2386, 8.2473, PDist.lognorm(((1.8898,), 0.2381, 0.2701))),
    ('pi', 'alexrashed/ml-wf-3-serve:0.37', 25000000.0):
        (0.2019, 4.5998, PDist.lognorm(((1.8239,), 0.2013, 0.2212))),
    ('pi', 'alexrashed/ml-wf-3-serve:0.37', 125000000.0):
        (0.1633, 5.5627, PDist.lognorm(((1.7236,), 0.1624, 0.2716))),
    ('tegra', 'alexrashed/ml-wf-1-pre:0.37', 1250000.0):
        (60.7136, 79.5575, PDist.lognorm(((0.1542,), 43.926, 23.8042))),
    ('tegra', 'alexrashed/ml-wf-1-pre:0.37', 12500000.0):
        (43.7871, 78.9973, PDist.lognorm(((0.2988,), 34.7842, 18.611))),
    ('tegra', 'alexrashed/ml-wf-1-pre:0.37', 25000000.0):
        (35.0163, 64.1116, PDist.lognorm(((0.1227,), 5.9188, 40.0595))),
    ('tegra', 'alexrashed/ml-wf-1-pre:0.37', 125000000.0):
        (16.6969, 37.9708, PDist.lognorm(((0.1307,), -10.0019, 34.4951))),
    ('tegra', 'alexrashed/ml-wf-2-train:0.37', 1250000.0):
        (216.8735, 406.7351, PDist.lognorm(((1.2725,), 216.8032, 2.4873))),
    ('tegra', 'alexrashed/ml-wf-2-train:0.37', 12500000.0):
        (45.5846, 108.5807, PDist.lognorm(((0.6943,), 44.6426, 5.8439))),
    ('tegra', 'alexrashed/ml-wf-2-train:0.37', 25000000.0):
        (36.6631, 59.8366, PDist.lognorm(((0.8827,), 36.4426, 2.051))),
    ('tegra', 'alexrashed/ml-wf-2-train:0.37', 125000000.0):
        (36.5853, 79.3900, PDist.lognorm(((0.7638,), 36.2463, 2.9114))),
    ('tegra', 'alexrashed/ml-wf-3-serve:0.37', 1250000.0):
        (0.2470, 0.3707, PDist.lognorm(((0.045,), -0.1412, 0.4273))),
    ('tegra', 'alexrashed/ml-wf-3-serve:0.37', 12500000.0):
        (0.1255, 0.1817, PDist.lognorm(((0.0097,), -1.0322, 1.186))),
    ('tegra', 'alexrashed/ml-wf-3-serve:0.37', 25000000.0):
        (0.0926, 0.1487, PDist.lognorm(((0.0835,), -0.0176, 0.1323))),
    ('tegra', 'alexrashed/ml-wf-3-serve:0.37', 125000000.0):
        (0.0666, 0.1146, PDist.lognorm(((0.0643,), -0.0752, 0.1638))),
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
