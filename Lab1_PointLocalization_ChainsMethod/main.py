import copy
import math

import matplotlib.pyplot as plt
import numpy


class Vertex:
    def __init__(self, x_coordinate, y_coordinate, n) -> None:
        self.in_list = None
        self.out_list = None

        self.x = x_coordinate
        self.y = y_coordinate
        self.n = n
        self.weight = 1
        self.neighbours = []

    def set_number(self, new_number):
        self.n = new_number

    def add_weight(self, val):
        self.weight += val

    def __lt__(self, other):
        return self.y < other.y or (self.y == other.y and self.x < other.x)

    def __str__(self):
        return str(f"{self.n}: ({self.x}; {self.y})")

    def __repr__(self):
        return str(f"{self.n}: ({self.x}; {self.y})")

    def create_in_out_lists(self):
        self.in_list = []
        self.out_list = []

        for neighbour in self.neighbours:
            if neighbour.n < self.n:
                self.in_list.append(neighbour)
            else:
                self.out_list.append(neighbour)

        self.in_list = sort_clockwise(self.in_list, self)
        self.in_list.reverse()

        self.out_list = sort_clockwise(self.out_list, self)

    def in_in_list(self, index):
        for vertex in self.in_list:
            if vertex.n == index:
                return True
        return False

    def in_out_list(self, index):
        for vertex in self.out_list:
            if vertex.n == index:
                return True
        return False

    def plot(self, ax):
        ax.scatter(self.x, self.y, color="red")

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        if self.x >= x_max:
            ax.set_xlim(x_min, math.ceil(self.x + 1))

        if self.y >= y_max:
            ax.set_ylim(y_min, math.ceil(self.y + 1))

        if self.x <= x_min:
            ax.set_xlim(math.ceil(self.x - 1), x_max)

        if self.y <= y_min:
            ax.set_xlim(math.ceil(self.y - 1), y_max)

        return ax


class Graph:
    def __init__(self):
        self.vertices = []
        self.weights = None

    def add_vertex(self, vertex):
        self.vertices.append(vertex)

    def add_edge(self, first, second):
        self.vertices[first].neighbours.append(self.vertices[second])
        self.vertices[second].neighbours.append(self.vertices[first])

    def init_weights(self):
        self.weights = []
        for i, _ in enumerate(self.vertices):
            self.weights.append([-1] * len(self.vertices))

        for vertex in self.vertices:
            for another_vertex in vertex.neighbours:
                self.weights[vertex.n][another_vertex.n] = 1

    def build_chains(self):
        self.sort_vertices()
        self.balancing_algorithm()
        weights = copy.deepcopy(self.weights)

        chains_count = 0
        for vertex in self.vertices[0].out_list:
            chains_count += weights[self.vertices[0].n][vertex.n]

        chains = []
        for _ in range(chains_count):
            chain = []
            current_vertex = self.vertices[0]
            while current_vertex != self.vertices[-1]:
                chain.append(current_vertex)
                j = 0
                while weights[current_vertex.n][current_vertex.out_list[j].n] < 1:
                    j += 1

                weights[current_vertex.n][current_vertex.out_list[j].n] -= 1
                weights[current_vertex.out_list[j].n][current_vertex.n] -= 1
                current_vertex = current_vertex.out_list[j]

            chain.append(self.vertices[-1])

            chains.append(chain)

        return chains

    def sort_vertices(self):
        self.vertices.sort()
        for i, vertex in enumerate(self.vertices):
            vertex.n = i

    def show_plot(self):
        xs = []
        ys = []
        labels = []

        for vertex in self.vertices:
            xs.append(vertex.x)
            ys.append(vertex.y)
            labels.append("(" + str(vertex.n) + ")")

        return self.plot_fig(xs, ys, labels)

    def plot_fig(self, xs, ys, labels):
        # plt.style.use('ggplot')

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

        ax.scatter(xs, ys, color="green")

        ax.set_xlim([min(xs) - 1, max(xs) + 1])
        ax.set_ylim([min(ys) - 1, max(ys) + 1])

        for vertex in self.vertices:
            for neighbour in vertex.neighbours:
                ax.plot([vertex.x, neighbour.x], [vertex.y, neighbour.y], "b")

                if self.weights is not None:
                    ax.annotate(self.weights[vertex.n][neighbour.n],
                                ((vertex.x + neighbour.x) / 2, (vertex.y + neighbour.y) / 2), color="purple")

        for i in range(len(xs)):
            ax.annotate(labels[i], (xs[i], ys[i]), xytext=(xs[i] - 0.025, ys[i] + 0.1))

        ax.set_ylabel("y", rotation=0, labelpad=20)
        ax.set_xlabel("x", rotation=0, labelpad=20)

        return fig, ax

    def balancing_algorithm(self):
        self.init_weights()

        self.prepare_in_out_lists()

        for index in range(1, len(self.vertices) - 1):
            current_vertex = self.vertices[index]
            leftest_vertex = current_vertex.out_list[0]
            w_in = 0
            for vertex in current_vertex.in_list:
                w_in += self.weights[vertex.n][current_vertex.n]

            w_out = 0
            for vertex in current_vertex.out_list:
                w_out += self.weights[current_vertex.n][vertex.n]

            if w_in > w_out:
                self.weights[current_vertex.n][leftest_vertex.n] += w_in - w_out
                self.weights[leftest_vertex.n][current_vertex.n] = self.weights[current_vertex.n][leftest_vertex.n]

        for index in range(len(self.vertices) - 1, 0, -1):
            current_vertex = self.vertices[index]
            leftest_vertex = current_vertex.in_list[0]
            w_in = 0
            for vertex in current_vertex.in_list:
                w_in += self.weights[vertex.n][current_vertex.n]

            w_out = 0
            for vertex in current_vertex.out_list:
                w_out += self.weights[current_vertex.n][vertex.n]

            if w_out > w_in:
                self.weights[leftest_vertex.n][current_vertex.n] += w_out - w_in
                self.weights[current_vertex.n][leftest_vertex.n] = self.weights[leftest_vertex.n][current_vertex.n]

    def prepare_in_out_lists(self):
        for i, vertex in enumerate(self.vertices):
            vertex.create_in_out_lists()


def point_localization(current_chains, point):
    is_on_left = False
    middle_pivot = int(len(current_chains) / 2 - 1)

    if len(current_chains) > 1:
        is_on_left = point_discrimination(current_chains[middle_pivot], point)

        if not is_on_left and point_discrimination(current_chains[middle_pivot + 1], point):
            return [current_chains[middle_pivot], current_chains[middle_pivot + 1]]

    if len(current_chains) == 1:
        if point_discrimination(current_chains[0], point):
            return [None, current_chains[0]]
        else:
            return [current_chains[0], None]

    if is_on_left:
        return point_localization(current_chains[:middle_pivot + 1], point)
    else:
        return point_localization(current_chains[middle_pivot + 1:], point)


def point_discrimination(chain, point):
    if len(chain) == 2:
        return is_point_left(chain[0], chain[1], point)
    length = len(chain)
    middle_pivot = int(length / 2)
    if point.y < chain[middle_pivot].y:
        return point_discrimination(chain[:middle_pivot + 1], point)
    else:
        return point_discrimination(chain[middle_pivot:], point)


def is_point_left(chain_point_1, chain_point_2, point):
    return ((chain_point_2.x - chain_point_1.x) * (point.y - chain_point_1.y) - (chain_point_2.y - chain_point_1.y) * (
            point.x - chain_point_1.x)) >= 0


def sort_clockwise(vertices: list, center_vertex):
    return sorted(vertices, key=lambda vertex: math.atan2(vertex.y - center_vertex.y,
                                                          vertex.x - center_vertex.x), reverse=True)


def read_file(path):
    file = open(path, "r")
    vertices_count = int(file.readline())
    edges_count = int(file.readline())

    new_graph = Graph()

    for i in range(vertices_count):
        line = str(file.readline())
        coordinates = line.split(sep=" ")
        new_graph.add_vertex(Vertex(float(coordinates[0]), float(coordinates[1]), i))

    for i in range(edges_count):
        line = str(file.readline())
        vertices = line.split(sep=" ")
        new_graph.add_edge(int(vertices[0]), int(vertices[1]))

    line = str(file.readline())
    coordinates = line.split(sep=" ")

    return new_graph, Vertex(float(coordinates[0]), float(coordinates[1]), -1)


def plot_chain(chain, ax, color, shift):
    for i in range(1, len(chain)):
        ax.plot([chain[i - 1].x + shift, chain[i].x + shift], [chain[i - 1].y + shift, chain[i].y + shift], color=color)

    return ax


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 22})

    graph, point = read_file("input.txt")

    chains = graph.build_chains()

    figure, axes = graph.show_plot()
    axes.set_title("Graph")
    axes = point.plot(axes)

    _, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    for i, chain in enumerate(chains):
        axes = plot_chain(chain, axes, numpy.random.rand(3, ), numpy.random.rand() * 0.11)

    axes.set_title("Chains")
    axes = point.plot(axes)
    _, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    location = point_localization(chains, point)
    for i, chain in enumerate(location):
        if chain is not None:
            axes = plot_chain(chain, axes, numpy.random.rand(3, ), 0)

    axes.set_title("Localization")
    axes = point.plot(axes)
    plt.show()
