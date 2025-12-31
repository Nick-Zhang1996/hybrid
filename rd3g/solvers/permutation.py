# permutation
from scipy import sparse
import numpy as np
Pinv1 = [538, 500, 415, 485, 516, 537, 505, 456, 534, 552, 521, 358, 426, 543, 425, 489, 539, 472, 515, 460, 540, 553, 531, 362, 541, 535, 423, 498, 533, 536, 513, 469, 532, 554, 529, 371, 555, 556, 443, 495, 557, 558, 544, 466, 559, 373, 545, 368, 551, 546, 432,
         499, 410, 547, 397, 470, 391, 562, 381, 372, 442, 542, 441, 477, 548, 471, 406, 448, 549, 560, 390, 350, 550, 568, 439, 569, 407, 564, 404, 563, 409, 561, 388, 374, 570, 15, 565, 4, 408, 571, 572, 301, 573, 574, 575, 319, 296, 576, 285, 14, 577, 312, 39, 311,
         578, 567, 89, 329, 295, 579, 294, 11, 580, 313, 48, 308, 581, 566, 98, 326, 582, 33, 292, 32, 49, 583, 46, 584, 585, 586, 96, 345, 587, 31, 599, 20, 50, 588, 278, 55, 589, 590, 114, 334, 84, 591, 73, 30, 279, 66, 120, 65, 277, 592, 104, 344, 83, 593, 82, 27,
         273, 67, 129, 62, 274, 594, 113, 341, 272, 595, 80, 596, 130, 275, 127, 276, 269, 597, 111, 598, 270, 152, 271, 136, 131, 263, 264, 181, 267, 266, 268, 160, 254, 262, 210, 151, 265, 253, 226, 196]
Pinv2 = [255, 176, 242, 175, 247, 256, 212, 147, 246, 248, 228, 192, 245, 251, 244, 171, 257, 153, 203, 144, 229, 258, 219, 189, 259, 260, 235, 168, 213, 261, 201, 141, 250, 197, 217, 186, 249, 252, 233, 165, 413, 481, 503, 452, 519, 354, 412, 483, 502, 454, 518,
         356, 419, 487, 509, 458, 525, 360, 421, 491, 511, 462, 527, 364, 428, 493, 393, 464, 377, 366, 430, 473, 395, 444, 379, 346, 435, 475, 400, 446, 384, 348, 437, 0, 402, 297, 386, 315, 281, 2, 35, 299, 85, 317, 283, 7, 37, 304, 87, 322, 288, 9, 42, 306, 92, 324,
         290, 16, 44, 51, 94, 330, 69, 18, 116, 53, 100, 332, 71, 23, 118, 58, 102, 337, 76, 25, 123, 60, 107, 339, 78, 132, 125, 177, 109, 156, 206, 134, 222, 179, 238, 158, 208, 139, 224, 184, 240, 163, 199, 145, 215, 190, 231, 169, 200, 143, 216, 188, 232, 167, 411,
         482, 414, 480, 501, 453, 504, 451, 517, 355, 520, 353, 417, 486, 416, 484, 507, 457, 506, 455, 523, 359, 522, 357, 418, 490, 420, 488, 508, 461, 510, 459, 524, 363, 526, 361, 424, 496, 422, 492]
Pinv3 = [514, 467, 512, 463, 530, 369, 528, 365, 427, 497, 429, 494, 375, 468, 394, 465, 376, 370, 378, 367, 433, 478, 431, 474, 398, 449, 396, 445, 382, 351, 380, 347, 434, 479, 436, 476, 399, 450, 401, 447, 383, 352, 385, 349, 440, 5, 438, 1, 405, 302, 403, 298,
         389, 320, 387, 316, 280, 6, 282, 3, 392, 303, 36, 300, 314, 321, 86, 318, 286, 12, 284, 8, 40, 309, 38, 305, 90, 327, 88, 323, 287, 13, 289, 10, 41, 310, 43, 307, 91, 328, 93, 325, 293, 21, 291, 17, 47, 56, 45, 52, 97, 335, 95, 331, 68, 22, 70, 19, 34, 57, 117,
         54, 99, 336, 101, 333, 74, 28, 72, 24, 121, 63, 119, 59, 105, 342, 103, 338, 75, 29, 77, 26, 122, 64, 124, 61, 106, 343, 108, 340, 81, 137, 79, 133, 128, 182, 126, 178, 112, 161, 110, 157, 154, 138, 207, 135, 115, 183, 223, 180, 155, 162, 239, 159, 211, 148,
         209, 140, 227, 193, 225, 185, 243, 172, 241, 164, 198, 149, 204, 146, 214, 194, 220, 191, 230, 173, 236, 170, 205, 150, 202, 142, 221, 195, 218, 187, 237, 174, 234, 166]
mPinv = Pinv1 + Pinv2 + Pinv3


def permute_sparse_symmetric(A, Pinv, upper_triangular_only=True):
    """
    Permutes a sparse matrix A using inverse permutation Pinv.
    Equivalent to C = PAP'

    Args:
        A: scipy.sparse.csc_matrix (The matrix to permute)
        Pinv: list or np.array (The inverse permutation vector from QDLDL)
        upper_triangular_only: bool (If True, returns only upper triangle, like symperm)

    Returns:
        scipy.sparse.csc_matrix
    """
    n = A.shape[0]
    Pinv = np.array(Pinv)

    # 1. Convert Pinv (Old -> New) to P (New -> Old)
    # SciPy slicing requires a list of 'which old row to put here'
    P = np.zeros(n, dtype=int)
    P[Pinv] = np.arange(n)

    # 2. Perform the symmetric permutation: A[P, :][:, P]
    # Optimization: For CSC, slice columns first, then rows.
    # This physically moves the elements to their new (row, col) positions.
    A_perm = A[:, P][P, :]

    # 3. Handle Triangularity (Critical for QDLDL context)
    if upper_triangular_only:
        # If A was only the upper triangle, A_perm now has elements
        # below the diagonal because rows/cols were swapped.
        # We must reflect them back to the upper triangle.

        # Symmetrize the result (summing duplicates if any exist, though unlikely here)
        # Note: We use abs() > 0 or similar logic if we want structure,
        # but technically we just need the values.
        # The robust way to get the upper tri of a symmetric permutation
        # of an originally upper-tri matrix:

        # 3a. Ensure we have the full symmetric values in the right spots
        # (This leverages the fact that A_perm is structurally symmetric
        # but stored with values potentially on the 'wrong' side of diagonal)
        A_perm = A_perm + A_perm.T

        # 3b. The diagonal was added twice, so subtract it once
        # (A easier way is usually strictly taking triu if we started full,
        # but since we started triu, we fix the diagonal)
        diag_indices = np.arange(n)
        A_perm[diag_indices, diag_indices] /= 2.0

        # 3c. Extract strictly upper triangle
        A_perm = sparse.triu(A_perm, format='csc')

    return A_perm


if __name__ == "__main__":
    # --- Example Usage ---

    # 1. Create a dummy symmetric matrix (Upper triangle only)
    # [ 4.  1.  0.]
    # [ 0.  3.  2.]
    # [ 0.  0.  5.]
    row = np.array([0, 0, 1, 1, 2])
    col = np.array([0, 1, 1, 2, 2])
    data = np.array([4., 1., 3., 2., 5.])
    KKT_reduced = sparse.csc_matrix((data, (row, col)), shape=(3, 3))

    # 2. Define Pinv (e.g., swap index 0 and 2)
    # Old 0 -> New 2
    # Old 1 -> New 1
    # Old 2 -> New 0
    Pinv = [2, 1, 0]

    # 3. Perform permutation
    KKT_permuted = permute_sparse_symmetric(KKT_reduced, Pinv)

    print("Original (Upper):\n", KKT_reduced.toarray())
    print("\nPermuted (Upper):\n", KKT_permuted.toarray())


def c_max(a, b):
    return a if a > b else b


def c_min(a, b):
    return a if a < b else b


def cumsum(p, c, n):
    """
    Computes the cumulative sum of c into p.
    p: Output column pointers (numpy array of size n+1)
    c: Input counts/workspace (numpy array of size n)
    n: Integer number of columns
    """
    if p is None or c is None:
        return -1

    nz = 0
    for i in range(n):
        p[i] = nz
        nz += c[i]
        c[i] = p[i]  # c is updated to act as the current pointer in the next steps

    p[n] = nz
    return nz


def permute_x(n, x, b, P):
    """
    Computes x = P * b
    x: Output array
    b: Input array
    P: Permutation vector
    """
    for j in range(n):
        x[j] = b[P[j]]


def permutet_x(n, x, b, P):
    """
    Computes x = P' * b (Inverse permutation)
    x: Output array
    b: Input array
    P: Permutation vector
    """
    for j in range(n):
        x[P[j]] = b[j]


def pinv(p, pinv_arr, n):
    """
    Computes the inverse of permutation p.
    p: Input permutation vector
    pinv_arr: Output inverse permutation vector
    """
    for k in range(n):
        pinv_arr[p[k]] = k


def symperm(A, pinv_arr, AtoC, w):
    """
    Computes symmetric permutation C = PAP' where A and C are symmetric 
    (upper part stored).

    Args:
        A: Input matrix A in CSC format (Upper triangular)
        pinv_arr: Inverse permutation vector (can be None)
        AtoC: Mapping vector from A indices to C indices (can be None)
        w: Workspace array (size n)
    Returns:
        C: Output matrix C in CSC format (Upper triangular)
    """
    Ap = A.indptr
    Ai = A.indices
    Ax = A.data

    n = A.shape[0]
    Cp = np.zeros_like(Ap)
    Ci = np.zeros_like(Ai)
    Cx = np.zeros_like(Ax)

    # 1. Count entries in each column of C
    for j in range(n):
        j2 = pinv_arr[j] if pinv_arr is not None else j

        # Iterate over column j of A
        for p in range(Ap[j], Ap[j+1]):
            i = Ai[p]

            if i > j:
                continue  # Skip lower triangular part of A

            i2 = pinv_arr[i] if pinv_arr is not None else i

            # Increment count for the column index in C (which is max(i2, j2))
            col_idx = c_max(i2, j2)
            w[col_idx] += 1

    # 2. Compute column pointers for C
    cumsum(Cp, w, n)

    # 3. Fill in indices and values
    for j in range(n):
        j2 = pinv_arr[j] if pinv_arr is not None else j

        for p in range(Ap[j], Ap[j+1]):
            i = Ai[p]

            if i > j:
                continue  # Skip lower triangular part of A

            i2 = pinv_arr[i] if pinv_arr is not None else i

            # Determine row and col in C
            # In C: Ci[q = w[c_max(i2, j2)]++] = c_min(i2, j2);
            col_idx = c_max(i2, j2)
            row_idx = c_min(i2, j2)

            q = w[col_idx]  # Get current pointer
            w[col_idx] += 1  # Post-increment the pointer

            Ci[q] = row_idx

            if Cx is not None and Ax is not None:
                Cx[q] = Ax[p]

            if AtoC is not None:
                AtoC[p] = q
    C = sparse.csc_matrix((Cx, Ci, Cp), shape=A.shape)
    return C


def update_A(Anz, Apermx, Anewx, AtoAperm):
    """
    Updates the values of the permuted matrix using the mapping.
    """
    for i in range(Anz):
        Apermx[AtoAperm[i]] = Anewx[i]
