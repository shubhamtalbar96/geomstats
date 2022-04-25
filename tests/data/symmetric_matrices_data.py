import random

from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from tests.data_generation import _VectorSpaceTestData


class SymmetricMatricesTestData(_VectorSpaceTestData):
    """Data class for Testing Symmetric Matrices"""

    space_args_list = [(n,) for n in random.sample(range(2, 5), 2)]
    n_points_list = random.sample(range(1, 5), 2)
    shape_list = [(n, n) for (n,), in zip(space_args_list)]
    n_vecs_list = random.sample(range(1, 5), 2)

    def belongs_test_data(self):
        smoke_data = [
            dict(n=2, mat=[[1.0, 2.0], [2.0, 1.0]], expected=True),
            dict(n=2, mat=[[1.0, 1.0], [2.0, 1.0]], expected=False),
            dict(
                n=3,
                mat=[[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
                expected=True,
            ),
            dict(
                n=2,
                mat=[[[1.0, 0.0], [0.0, 1.0]], [[1.0, -1.0], [0.0, 1.0]]],
                expected=[True, False],
            ),
        ]
        return self.generate_tests(smoke_data)

    def basis_test_data(self):
        smoke_data = [
            dict(n=1, basis=[[[1.0]]]),
            dict(
                n=2,
                basis=[
                    [[1.0, 0.0], [0, 0]],
                    [[0, 1.0], [1.0, 0]],
                    [[0, 0.0], [0, 1.0]],
                ],
            ),
        ]
        return self.generate_tests(smoke_data)

    def expm_test_data(self):
        smoke_data = [
            dict(mat=[[0.0, 0.0], [0.0, 0.0]], expected=[[1.0, 0.0], [0.0, 1.0]])
        ]
        return self.generate_tests(smoke_data)

    def powerm_test_data(self):
        smoke_data = [
            dict(
                mat=[[1.0, 2.0], [2.0, 3.0]],
                power=1.0,
                expected=[[1.0, 2.0], [2.0, 3.0]],
            ),
            dict(
                mat=[[1.0, 2.0], [2.0, 3.0]],
                power=2.0,
                expected=[[5.0, 8.0], [8.0, 13.0]],
            ),
        ]
        return self.generate_tests(smoke_data)

    def dim_test_data(self):

        smoke_data = [dict(n=1, dim=1), dict(n=2, dim=3), dict(n=5, dim=15)]

        return self.generate_tests(smoke_data, [])

    def to_vector_test_data(self):
        smoke_data = [
            dict(n=1, mat=[[1.0]], vec=[1.0]),
            dict(
                n=3,
                mat=[[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
                vec=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ),
            dict(
                n=3,
                mat=[
                    [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
                    [[7.0, 8.0, 9.0], [8.0, 10.0, 11.0], [9.0, 11.0, 12.0]],
                ],
                vec=[
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                ],
            ),
        ]
        return self.generate_tests(smoke_data)

    def from_vector_test_data(self):
        smoke_data = [
            dict(n=1, vec=[1.0], mat=[[1.0]]),
            dict(
                n=3,
                vec=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                mat=[[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
            ),
            dict(
                n=3,
                vec=[
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                ],
                mat=[
                    [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
                    [[7.0, 8.0, 9.0], [8.0, 10.0, 11.0], [9.0, 11.0, 12.0]],
                ],
            ),
        ]
        return self.generate_tests(smoke_data)

    def basis_belongs_test_data(self):

        return self._basis_belongs_test_data(self.space_args_list)

    def basis_cardinality_test_data(self):
        return self._basis_cardinality_test_data(self.space_args_list)

    def projection_belongs_test_data(self):
        return self._projection_belongs_test_data(
            self.space_args_list, self.shape_list, self.n_points_list
        )

    def to_tangent_is_tangent_test_data(self):
        return self._to_tangent_is_tangent_test_data(
            SymmetricMatrices,
            self.space_args_list,
            self.shape_list,
            self.n_vecs_list,
        )

    def random_tangent_vec_is_tangent_test_data(self):
        return self._random_tangent_vec_is_tangent_test_data(
            SymmetricMatrices, self.space_args_list, self.n_vecs_list
        )

    def random_point_belongs_test_data(self):
        smoke_space_args_list = [(1,), (2,), (3,)]
        smoke_n_points_list = [1, 1, 10]

        return self._random_point_belongs_test_data(
            smoke_space_args_list,
            smoke_n_points_list,
            self.space_args_list,
            self.n_points_list,
        )

    def to_tangent_is_projection_test_data(self):
        return self._to_tangent_is_projection_test_data(
            SymmetricMatrices,
            self.space_args_list,
            self.shape_list,
            self.n_vecs_list,
        )

    def random_point_is_tangent_test_data(self):
        return self._random_point_is_tangent_test_data(
            self.space_args_list, self.n_points_list
        )
