# TODO: needs debugging

_SVC = namedtuple(typename='SVC', field_names='predict n_sv')

def fit_SVM(X, y, kernel='linear', degree=None, C=None):
    # ===== Preparation ===== 
    N = y.size

    X_has_dummy_variable = np.array_equal(X[:, 0], np.ones(N))
    if X_has_dummy_variable: X = X[:, 1:]
    
    # Define kernel function
    if kernel == 'poly' and degree is None:
        raise ValueError("if polynomial kernel is used, 'degree' must be specified")
    K = ({'linear': lambda xn, xm: xn@xm,
          'poly'  : lambda xn, xm: np.exp(degree * np.log(1 + xn@xm)),
          'rbf'   : lambda xn, xm: np.exp(-np.square(norm(xn - xm)))}
         .get(kernel))
    if not K:
        raise ValueError("unsupported kernel type: '{}'".format(kernel))
    
    # ===== Quadratic programming part =====    
    # Solver won't give back alphas for non-support vectors exactly equaling zero.
    # For this reason we're placing a threshold below which we'll consider a value
    # being zero.
    zero_value_threshold = 1e-6
    
    quad_coefs = np.empty((N, N))
    for n in range(N):
        for m in range(N):
            quad_coefs[n, m] = y[n] * y[m] * K(X[n], X[m])
    
    # Adjust QP conditions if soft-margin is applied
    G = -np.identity(N)
    h = np.zeros(N)
    if C:
        G = np.vstack((G, np.identity(N)))
        h = np.append(h, np.full(N, fill_value=C))

    P = cvxopt.matrix(quad_coefs)
    q = cvxopt.matrix(-np.ones(N))
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)
    A = cvxopt.matrix(y, (1, N))
    b = cvxopt.matrix(0.0)

    cvxopt.solvers.options['show_progress'] = False
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    alpha = np.array(solution['x']).reshape(N)
    
    # ===== Getting final hypothesis =====
    # Caution: the realization is very inefficient
    sv_idxs = np.where(alpha > zero_value_threshold)
    
    a_sv = alpha[sv_idxs]
    y_sv = y[sv_idxs]
    X_sv = X[sv_idxs]
    
    @np.vectorize
    def kernel_dot(x):
        return sum(a_sv[n] * y_sv[n] * K(X_sv[n], x) for n in range(a_sv.size))
    
    b = y_sv[0] - kernel_dot(X_sv[0])
    
    @np.vectorize
    def predict(x):
        return np.sign(kernel_dot(x) + b)

    return _SVC(predict=predict,
                n_sv=a_sv.size)

