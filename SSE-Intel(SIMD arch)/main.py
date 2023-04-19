import time
import numpy as np
import trimesh
import pyglet
import os
from assignment import transform_vertices


def load_the_file():
    filename = 'teapot.obj'
    url = "https://graphics.stanford.edu/courses/cs148-10-summer/as3/code/as3/teapot.obj"
    if not os.path.exists(filename):
        os.system(f'wget {url}')
        print("loaded")
    return trimesh.load(filename)


def do_verification(vertex_buffer_for_verification: list, transformation_matrix: list, vertex_buffer):
    vertex_buffer_for_verification = [[vertex_buffer_for_verification[0],
                                       vertex_buffer_for_verification[1],
                                       vertex_buffer_for_verification[2],
                                       1.0] for i in range(0, len(vertex_buffer) // 3)]

    matrix = [[transformation_matrix[i], transformation_matrix[4 + i], \
                              transformation_matrix[8 + i], transformation_matrix[12 + i]] for i in range(0, 4)]
    start_time = time.time()
    for i in range(0, len(vertex_buffer_for_verification)):
        row = [vertex_buffer_for_verification[i][0], vertex_buffer_for_verification[i][1], \
               vertex_buffer_for_verification[i][2], vertex_buffer_for_verification[i][3]]
        for j in range(0, 4):
            vertex_buffer_for_verification[i][j] = row[0] * matrix[j][0] + \
                                                   row[1] * matrix[j][1] + \
                                                   row[2] * matrix[j][2] + \
                                                   row[3] * matrix[j][3]
    end_time = time.time()
    print("EXECUTION TIME OF THE SIMPLEST PYTHON CODE: ", end_time - start_time, " seconds.")
    for i in range(0, len(vertex_buffer_for_verification)):
        if abs(vertex_buffer_for_verification[i][0] - vertex_buffer[i * 3 + 0]) < 1.0 or \
                abs(vertex_buffer_for_verification[i][1] - vertex_buffer[i * 3 + 1]) < 1.0 or \
                abs(vertex_buffer_for_verification[i][2] - vertex_buffer[i * 3 + 2]) < 1.0:
            print("VERIFICATION FAILED")
            return False
    print("VERIFIED")
    return True


if __name__ == "__main__":
    transformation_matrix = np.array([
        8.0, 0.0, 0.0, 0.0,
        0.0, 8.0, 0.0, 0.0,
        0.0, 0.0, 8.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    ], dtype=np.float32)
    model = load_the_file()
    model.show()
    # Extract vertices to a float buffer:
    vertex_buffer = np.ravel(model.vertices.view(np.ndarray)).astype(np.float32)
    vertex_buffer_for_verification = np.ravel(model.vertices.view(np.ndarray)).astype(np.float32)
    # Transform the vertex buffer with your C processing function:
    start_time = time.time()
    transform_vertices(vertex_buffer, transformation_matrix)
    end_time = time.time()
    print("EXECUTION OF C WITH INTRINSICS: ", end_time - start_time, " seconds.")
    # Reassign the transformed vertices to the model object and visualize it:
    if do_verification(vertex_buffer_for_verification, transformation_matrix, vertex_buffer):
        model.vertices = vertex_buffer.reshape((-1, 3))
        model.show()
    # BCS my verification fails
    model.vertices = vertex_buffer.reshape((-1, 3))
    model.show()
