import matplotlib.pyplot as plt
import numpy as np
from pyparsing import col
from pysfrl.sim.parameters import DataIndex as Index
from pysfrl.sim.new_simulator import NewSimulator


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["animation.html"] = "jshtml"
COLOR_LIST = ["black", "navy", "blue", "red", "darkgreen", "gray", "gold", "silver", "orange", "deeppink"]


class PlotGenerator(object):

    @staticmethod
    def generate_sim_result_plot(xy_range, sim: NewSimulator):
        sub_plots = PlotGenerator.generate_sub_plots(xy_range)
        sub_plots = PlotGenerator.plot_trajectory(sub_plots, sim.peds_states)
        sub_plots = PlotGenerator.plot_obstacles(sub_plots, sim.get_obstacles())
        return sub_plots

    @staticmethod
    def generate_sim_result_comparsion(xy_range, sim: NewSimulator, gt_path):
        sub_plots = PlotGenerator.generate_sub_plots(xy_range)
        sub_plots = PlotGenerator.plot_obstacles(sub_plots, sim.get_obstacles())
        sub_plots = PlotGenerator.plot_trajectory(sub_plots, sim.peds_states)


    # plot 틀을 생성
    @staticmethod
    def generate_sub_plots(xy_range):
        # trajectory graph margin
        x_min, x_max, y_min, y_max = xy_range
        margin = 2.0
        x_min, y_min = x_min - margin, y_min - margin
        x_max, y_max = x_max + margin, y_max + margin

        fig, ax = plt.subplots()        
        ax.grid(linestyle="dotted")
        ax.set_aspect("equal")
        ax.margins(2.0)
        ax.set_axisbelow(True)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        
        ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
        return fig, ax

    # 시뮬레이션 결과 생성
    @staticmethod
    def plot_trajectory(sub_plots, peds_states):
        fig, ax = sub_plots
        num_peds = len(peds_states[0])
        for ped_id in range(0, num_peds):
            states = peds_states[:,ped_id]
            px, py = states[:, Index.px.index], states[:, Index.py.index]
            visible = states[:, Index.visible.index] == 1
            color = COLOR_LIST[ped_id % len(COLOR_LIST)]
            ax.plot(px[visible], py[visible], "-o", markersize=0.05, color=color, label=ped_id)
            ax.legend()
        return fig, ax
            
    @staticmethod
    def plot_obstacles(sub_plots, obstacles):
        fig, ax = sub_plots        
        for s in obstacles:
            ax.plot(s[:, 0], s[:, 1], "-o", color="black", markersize=0.5)
        return fig, ax
