import torch


def h00(t):
    return 2 * t ** 3 - 3 * t ** 2 + 1


def h10(t):
    return t ** 3 - 2 * t ** 2 + t


def h01(t):
    return -2 * t ** 3 + 3 * t ** 2


def h11(t):
    return t ** 3 - t ** 2


class HermeticSpline:
    """
    Hermetic Spline class
    """

    def __init__(self, x, y):
        assert x.shape == y.shape, "x and y need to have the same shape."
        m = torch.diff(y, dim=-1) / torch.diff(x, dim=-1)
        self.m = torch.cat((m[:, :1], (m[:, 1:] + m[:, :-1]) / 2, m[:, -1:]), dim=-1)  # [num_waypoints]
        self.x = x  # distances [bs, num_waypoints]
        self.y = y  # coordinates on axis [bs, num_waypoints]

    def calc(self, t):
        """
        t is a batch of vectors of new distances [bs, num_nodes]
        """
        assert torch.logical_and(
            (t >= self.x[:, :1]), (t <= self.x[:, -1:])
        ).all(), "The evaluated distances have to be in the range of waypoint distances, we are interpolating, not extrapolating!"

        I = torch.searchsorted(self.x[:, 1:].contiguous(), t)
        dx = torch.gather(self.x, index=I + 1, dim=-1) - torch.gather(self.x, index=I, dim=-1)
        u = (t - torch.gather(self.x, index=I, dim=-1)) / dx

        first_term = h00(u) * torch.gather(self.y, index=I, dim=-1)
        second_term = h10(u) * torch.gather(self.m, index=I, dim=-1) * dx

        third_term = h01(u) * torch.gather(self.y, index=I + 1, dim=-1)
        fourth_term = h11(u) * torch.gather(self.m, index=I + 1, dim=-1) * dx
        return first_term + second_term + third_term + fourth_term


class Spline2D:
    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)  # [bs, num_waypoints]

        self.sx = HermeticSpline(self.s, x)
        self.sy = HermeticSpline(self.s, y)

    def __calc_s(self, x, y):
        assert x.shape == y.shape, "x and y need to have the same shape"
        bs = x.shape[0]
        dx = torch.diff(x, dim=-1)
        dy = torch.diff(y, dim=-1)

        self.ds = torch.sqrt(dx.pow(2) + dy.pow(2))
        s = torch.cat((torch.zeros(bs, 1).to(self.ds), torch.cumsum(self.ds, dim=-1)), dim=-1)
        self.travel_dist = s[:, -1]  # [bs]
        return s

    def calc_position(self, s):
        """
        calc position
        """
        x = self.sx.calc(s)
        y = self.sy.calc(s)

        return x, y


def calc_2d_spline_interpolation(x, y, num_nodes=100):
    """
    Calc 2d spline course with interpolation
    :param x: x positions [bs, num_waypoints]
    :param y: y positions [bs, num_waypoints]
    :param num_nodes: number of path points
    :return:
        - x     : interpolated x positions [bs, num_nodes]
        - y     : interpolated y positions [bs, num_nodes]
    """
    sp = Spline2D(x, y)
    s = torch.linspace(0, 1, num_nodes).to(x)
    s = s.unsqueeze(0) * sp.travel_dist.unsqueeze(1)
    r_x, r_y = sp.calc_position(s)
    return r_x, r_y


def interpolate_waypoints_using_splines(waypoints, num_nodes=20):
    """
    Calc 2d spline course with interpolation
    :param waypoints: waypoints [bs, num_waypoints, 2]
    :param num_nodes: number of path points
    :return:
        - trajectory : interpolated waypoints [bs, num_nodes, 2]
    """
    x = waypoints[:, :, 0]
    y = waypoints[:, :, 1]

    rx, ry = calc_2d_spline_interpolation(x, y, num_nodes)
    trajectory = torch.cat((rx.unsqueeze(-1), ry.unsqueeze(-1)), dim=-1)
    return trajectory
