import ConfigSpace

def main():
    cs = get_hyperparameter_search_space()


def get_hyperparameter_search_space():
    C = ConfigSpace.UniformFloatHyperparameter("C", 0.02, 32, log=True, default_value=1.0)
    kernel = ConfigSpace.CategoricalHyperparameter(name='kernel', choices=['rbf', 'poly', 'sigmoid'], default_value='rbf')
    degree = ConfigSpace.UniformIntegerHyperparameter('degree', 2, 5, default_value=3)
    gamma = ConfigSpace.UniformFloatHyperparameter('gamma', 3e-05, 8, log=True, default_value=0.1)
    coef0 = ConfigSpace.UniformFloatHyperparameter('coef0', -1, 1, default_value=0)
    shrinking = ConfigSpace.CategoricalHyperparameter('shrinking', ['True', 'False'], default_value='True')
    tol = ConfigSpace.UniformFloatHyperparameter('tol', 1e-5, 1e-1, default_value=1e-3, log=True)
    max_iter = ConfigSpace.UnParametrizedHyperparameter('max_iter', -1)

    cs = ConfigSpace.ConfigurationSpace()
    cs.add([C, kernel, degree, gamma, coef0, shrinking, tol, max_iter])
    # Hyperparameter child is conditional on the parent hyperparameter being equal to value
    degree_depends_on_poly = ConfigSpace.EqualsCondition(degree, kernel, 'poly')
    # Hyperparameter child is conditional on the parent hyperparameter being in a set of values
    coef0_condition = ConfigSpace.InCondition(coef0, kernel, ['poly', 'sigmoid'])
    cs.add(degree_depends_on_poly)
    cs.add(coef0_condition)
    return cs

main()