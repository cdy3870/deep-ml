def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    # Your code here
    trans_mat = []
    for i in range(len(a[0])):
        new_row = []
        for j in range(len(a)):
            new_row.append(a[j][i])
        trans_mat.append(new_row)

    return trans_mat