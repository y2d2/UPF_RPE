import unittest
import Code.UtilityCode.Transformation_Matrix_Fucntions as TMF

import numpy as np
import matplotlib.pyplot as plt
class MyTestCase(unittest.TestCase):
    def test_get_t(self):
        t = [1,1,1,5]
        T = TMF.transformation_matrix_from_4D_t(t)
        t_ = TMF.get_4D_t_from_matrix(T)
        print(t, t_)