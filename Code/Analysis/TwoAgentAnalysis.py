import numpy as np
import pickle as pkl
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import time

from matplotlib import rcParams


class TwoAgentAnalysis:
    def __init__(self, result_folders):
        if type(result_folders) is not list:
            result_folders = [result_folders]
        self.result_folders = result_folders
        self.results = None
        self.data = None
        self.dfs = []
        self.df = None
        self.percent_to_load = 100.
        # LOS Paper names
        # self.names = { "algebraic": "Algebraic", "WLS": "WLS", "NLS": "NLS", "upf": "UPF (ours, proposed)", "losupf": r"UPF $\tilde{w}$ $s_{LOS}$ (ours)", "nodriftupf": r"UPF $\tilde{w}$ $\theta_d$ (ours)"}
        # self.names = { "algebraic": "Algebraic", "WLS": "WLS", "NLS": "NLS", "upf": "upf", "losupf": "Ours, proposed", "nodriftupf": "Ours, without pseudo-state", "QCQP": "QCQP",
        #                "losupf | resampling_factor = 0.1 | sigma_uwb_factor = 2.0" : "losupf|resampling_factor=0.1|sigma_uwb_factor=2.0"}

        self.names = {"losupf|resample_factor=0.1|sigma_uwb_factor=2.0": "UPF, RF=0.1, $\sigma_uwb_f$=2.0",
                      "losupf|resample_factor=0.1|sigma_uwb_factor=1.0": "UPF, RF=0.1, $\sigma_uwb_f$=1.0",
                      "losupf|resample_factor=0.5|sigma_uwb_factor=2.0": "UPF, RF=0.5, $\sigma_uwb_f$=2.0",
                      "NLS|horizon=10": "NLS, horizon=10",
                      #"NLS|horizon=100",
                      "algebraic|horizon=10": "Algebraic, horizon=10",
                      "algebraic|horizon=100": "Algebraic, horizon=10",
                      "QCQP|horizon=10": "QCQP, horizon=10",
                      "QCQP|horizon=100": "QCQP, horizon=10"}

        # NLOS paper names
        # self.names = { "algebraic": "Algebraic", "WLS": "WLS", "NLS": "NLS", "upf": "NLOS UPF (ours)", "losupf":  "UPF (ours)" , "nodriftupf": r"UPF $\tilde{w}$ $\theta_d$ (ours)"}
        self.y_label = {
            "error_x_relative": "Error on the position ($\epsilon_{p}$) [m]",
            "error_h_relative": "Error on the orientation ($\epsilon_{\\theta}$) [rad]",
            "calculation_time": "Calculation time [s]"
        }

    # -----------------------
    # Loading functions:
    # -----------------------
    def load_results(self):
        self.results = {}
        self.data = {}
        for result_folder in self.result_folders:
            n_files = len(os.listdir(result_folder))
            file_nr = 0.
            for file in os.listdir(result_folder):
                if int(file_nr/n_files*100.) > self.percent_to_load:
                    return
                file_nr +=1
                if file.endswith(".pkl"):
                    with open(result_folder + "/" + file, "rb") as f:
                        try:
                            data = pkl.load(f)
                            print("Loading " + str(int(file_nr/n_files*100.)), "%: " + result_folder + "/" + file)
                        except EOFError: print("!!!!!!!!! Could not open: ", result_folder + "/" + file + " !!!!!!!!!")
                    f.close()
                    if "numerical_data" not in data:
                        print("Reformating the data for analysis " + file + " ...")
                        data = self.reformat_data(data)
                        with open(result_folder + "/" + file, "wb") as f:
                            pkl.dump(data, f)
                        f.close()
                    # self.data[file] = data
                    self.data[file] = "Loaded"

                    # if "analysis" not in data:
                    #     print("Starting statistical analysis of " + file + " ...")
                    #     data = self.calculate_statistics(data)
                    #     with open(self.result_folder + "/" + file, "wb") as f:
                    #         pkl.dump(data, f)
                    #     f.close()
                    # self.results[file] = data["analysis"]

                    # if "panda_date" not in data:
                    self.reformat_data_to_pandas(data)

    def reformat_data(self, data):
        data["numerical_data"] = {}
        result = {}
        for sim in data:
            if sim != "parameters" and sim != "analysis" and sim != "numerical_data":
                for method in data[sim]:
                    for drone_name in data[sim][method]:
                        if method not in data["numerical_data"]:
                            data["numerical_data"][method] = {}
                            result[method] = {}
                        for variable in data[sim][method][drone_name]:
                            if variable not in data["numerical_data"][method]:
                                data["numerical_data"][method][variable] = {}
                                result[method][variable] = []
                            result[method][variable].append(data[sim][method][drone_name][variable])
        for method in data["numerical_data"]:
            for variable in data["numerical_data"][method]:
                res = np.array(result[method][variable]).astype(float)
                data["numerical_data"][method][variable] = res
        return data

    def calculate_statistics(self, data):
        data["analysis"] = {}
        for method in data["numerical_data"]:
            data["analysis"][method] = {}
            for variable in data["numerical_data"][method]:
                data["analysis"][method][variable] = {}
                res = data["numerical_data"][method][variable]
                data["analysis"][method][variable]["mean"] = np.mean(res, axis=0)
                data["analysis"][method][variable]["sigma"] = np.std(res, axis=0)
                data["analysis"][method][variable]["median"] = np.median(res, axis=0)
                upper_q = np.percentile(res, 75, axis=0)
                lower_q = np.percentile(res, 25, axis=0)
                data["analysis"][method][variable]["upper_quartile"] = upper_q
                data["analysis"][method][variable]["lower_quartile"] = lower_q
                data["analysis"][method][variable]["iqr"] = upper_q - lower_q
                upper_w = res.copy()
                upper_w[res > upper_q + 1.5 * (upper_q - lower_q)] = np.nan
                data["analysis"][method][variable]["upper_whisker"] = np.nanmax(upper_w, axis=0)
                lower_w = res.copy()
                lower_w[res < lower_q - 1.5 * (upper_q - lower_q)] = np.nan
                data["analysis"][method][variable]["lower_whisker"] = np.nanmin(lower_w, axis=0)

                data["analysis"][method][variable]["max"] = np.nanmax(res, axis=0)
                data["analysis"][method][variable]["min"] = np.nanmin(np.array(res), axis=0)
        return data

    # -----------------------
    # Panda DF functions:
    # -----------------------
    def reformat_data_to_pandas(self, data):
        # data["panda_data"] = {}
        # result = { }
        # for sim in data:
        #     if sim != "parameters" and sim != "analysis" and sim != "numerical_data":
        #         for drone_name in data[sim]:
        #             for method in data[sim][drone_name]:
        #                 if method not in result:
        #                     result[method]={}
        #                 for variable in data[sim][drone_name][method]:
        #                     if variable not in result[method]:
        #                         result[method][variable]=[]
        #                     result[method][variable].append(data[sim][drone_name][method][variable])
        for method in data["numerical_data"]:
            for variable in data["numerical_data"][method]:
                res = np.array(data["numerical_data"][method][variable]).astype(float)
                df = pd.DataFrame(res).assign(Variable=variable,
                                              Method=method,
                                              Sigma_dv=data["parameters"]["sigma_dv"],
                                              Sigma_dw=data["parameters"]["sigma_dw"],
                                              Sigma_uwb=data["parameters"]["sigma_uwb"],
                                              Type=data["parameters"]["type"],
                                              Frequency=data["parameters"]["frequency"])
                df = pd.melt(df, id_vars = ['Variable', 'Method', 'Sigma_dv', 'Sigma_dw', 'Sigma_uwb', "Type", "Frequency"],
                var_name = ["Time"])
                self.dfs.append(df)

            # df_var = pd.concat([dfs[method][variable] for variable in dfs[method]])
            # df_var = pd.melt(df_var, id_vars=['Variable', 'Method', 'Sigma_dv', 'Sigma_uwb'], var_name=["Time [s]"])  # MELT
            # dfs[method] = df_var

    def create_panda_dataframe(self):
        if not self.dfs:
            self.load_results()
        # dfs = pd.concat(self.dfs)
        # self.df = pd.concat(self.dfs)
        return

    # -----------------------
    # delete functions:
    # -----------------------
    def delete_data(self):
        self.delete_data_from_files("analysis")
        self.delete_data_from_files("numerical_data")

    def delete_data_from_files(self, name):
        if name not in ["analysis", "numerical_data"]:
            print("Name not recognized")
            return
        for result_folder in self.result_folders:
            n_files = len(os.listdir(result_folder))
            file_nr = 0.
            for file in os.listdir(result_folder):
                file_nr += 1
                if file.endswith(".pkl"):
                    with open(result_folder + "/" + file, "rb") as f:
                        try:
                            print("Deleting " + name + " | " + str(int(file_nr / n_files * 100.)),
                                  "%: " + result_folder + "/" + file)
                            data = pkl.load(f)
                            if name in data:
                                del data[name]
                                with open(result_folder + "/" + file, "wb") as f:
                                    pkl.dump(data, f)
                                f.close()
                        except EOFError:
                            print("!!!!!!!!! Could not open: ", result_folder + "/" + file + " !!!!!!!!!")
                    f.close()

    # -----------------------
    # Support plotting functions:
    # -----------------------
    def remove_x_ticks(self, g, x_order, unit=""):
        # g.tick_params(bottom=False)
        g.set_axis_labels()

        if len(x_order) > 1:
            g.set_xticklabels(labels=[str(x) + unit for x in x_order])
        else:
            g.set_xticklabels(labels=[""])

        g.set_axis_labels("", "")
        g.set_titles("")
        # g.tick_params(bottom=False)

    def set_labels(self, g):
        smallest_x = 0
        smallest_y = 100
        x_values = []
        y_values = []
        for ax in g.axes_dict:
            if ax[0] not in x_values:
                x_values.append(ax[0])
            if ax[1] not in y_values:
                y_values.append(ax[1])
            if ax[0] > smallest_x:
                smallest_x = ax[0]
            if ax[1] < smallest_y:
                smallest_y = ax[1]

        for sigma_x in x_values:
            if g.axes_dict[(sigma_x, smallest_y)] != None:
                g.axes_dict[(sigma_x, smallest_y)].set_ylabel("VIO std: $\\sigma_v $= " + str(sigma_x),
                                                              fontdict={'fontsize': rcParams['axes.labelsize']})
        for sigma_y in y_values:
            if g.axes_dict[(smallest_x, sigma_y)] != None:
                g.axes_dict[(smallest_x, sigma_y)].set_xlabel("UWB std: $\\sigma_d $= " + str(sigma_y),
                                                              fontdict={'fontsize': rcParams['axes.labelsize']})


    def set_legend(self, g, methods_order, methods_legend = {}):
        if methods_legend == {}:
            g.add_legend()
            return

        legend_data = g._legend_data
        new_legend_data = {}

        # print(legend_data)
        for name in methods_order:
            if name in methods_legend:
                new_legend_data[methods_legend[name]] = legend_data[name]
            else:
                new_legend_data[name] = legend_data[name]
        g.add_legend(legend_data=new_legend_data)


    def print_statistics(self, methods, variable, df):
        for method in methods:
            print(method, variable, df[df["Method"] == method]["value"].mean(), " pm ",
                  df[df["Method"] == method]["value"].std(), "; median: ",
                  df[df["Method"] == method]["value"].median())

    def filter_methods(self, methods, sigma_uwb, sigma_v, frequencies, start_time):
        self.create_panda_dataframe()

        dfs = [df_i.loc[(df_i["Sigma_uwb"].isin(sigma_uwb)) &
                         (df_i["Sigma_dv"].isin(sigma_v)) &
                         (df_i["Time"] >  df_i["Frequency"] * start_time) &
                         (df_i["Frequency"].isin(frequencies))
                         ] for df_i in self.dfs]

        df = pd.concat(dfs)
        if not methods:
            methods = self.df["Method"].unique()

        # df = self.df.loc[(self.df["Method"].isin(methods)) &
        #                  (self.df["Sigma_uwb"].isin(sigma_uwb)) &
        #                  (self.df["Sigma_dv"].isin(sigma_v)) &
        #                  (self.df["Time"] > start_time_index) &
        #                  (self.df["Frequency"].isin(frequencies))
        #                  ]
        return df, methods

    # -----------------------
    # Plotting functions:
    # -----------------------
    def boxplots(self, sigma_uwb=[1.0, 0.1], sigma_v=[0.1,0.01],  frequencies = [1.0,10.0], start_time=0,
                 methods_order=None, methods_color=None, methods_legend = {}, x_variable = "Frequency",
                 x_order = None, save_fig=False, save_name="boxplot"):
        if x_order is None:
            x_order = frequencies

        unit = ""
        if x_variable == "Frequency":
            unit = "Hz"

        if methods_order is None:
            methods_order = []
        method_df, methods_order = self.filter_methods(methods_order, sigma_uwb, sigma_v, frequencies, start_time)
        gs = []
        for variable in self.y_label:
            df = method_df.loc[(method_df["Variable"] == variable)]
            g = sns.catplot(data=df, kind='box', col='Sigma_uwb', row="Sigma_dv", y='value', x=x_variable, hue='Method',
                                    dodge=True, aspect=0.65, order= x_order, palette=methods_color, hue_order=methods_order,
                            legend=False)
            self.remove_x_ticks(g, x_order, unit)
            self.set_labels(g)
            self.print_statistics(methods_order, variable, df)

            if variable == "calculation_time":
                self.set_legend(g,methods_order, methods_legend)
                for ax in g.axes_dict:
                    g.axes_dict[ax].set_yscale("log")

            if variable == "error_x_relative":
                for ax in g.axes_dict:
                    g.axes_dict[ax].set_yscale("log")

            g.fig.subplots_adjust(top=0.9)
            g.fig.suptitle(self.y_label[variable])
            if save_fig:
                g.fig.savefig(self.result_folders[0] + "/" + save_name + "_" + variable + ".png")
            gs.append(g)

    #------------------------
    # Time analysis
    #------------------------

    def time_analysis(self,sigma_uwb =0.1, sigma_v=0.1, frequency=10.0, start_time=0,
                      methods_order=[], methods_color=None, methods_legend = {},
                        save_fig=False, save_name="time_plot"):
        method_df, methods_order = self.filter_methods(methods_order, [sigma_uwb], [sigma_v], [frequency], start_time_index)


    def calculation_time(self, save_fig=False):
        if self.df is None:
            self.create_panda_dataframe()
        # method_colors = {"NLS": "tab:blue", "losupf": "tab:green", "nodriftupf": "tab:red", "algebraic": "tab:orange", "NLS_p":"tab:purple"}
        # order = ["losupf", "nodriftupf", "algebraic", "NLS"]
        # order = ["NLS", "NLS_p"]
        # plt.figure(figsize=(8, 4))
        fig, axes = plt.subplots(1, 1, figsize=(8, 4))
        # axes_i = 0
        for i, variable in enumerate(["calculation_time"]):  #self.y_label:
            df = self.df.loc[~self.df["Method"].isin(["slam", "WLS", "upf"]) & (self.df["Variable"] == variable)]
            methods = df["Method"].unique()
            method_means = []  # to store mean values for each method
            time_points = df["Time"].unique()
            time_points = time_points[:210]

            for method in methods:
                method_time_values = []  # to store values at each time point for a specific method
                for time_point in time_points:
                    mean_value = df[(df["Method"] == method) & (df["Time"] == time_point)]["value"].mean()
                    method_time_values.append(mean_value)

                method_means.append({"Method": method, "TimeValues": method_time_values})
                print("For Method:", method, variable, "Average over all conditions at each time point:",
                      method_time_values)

            # Create a new DataFrame with average values and time points
            avg_time_df = pd.DataFrame({"Time": time_points})

            for method_mean in method_means:
                avg_time_df[method_mean["Method"]] = method_mean["TimeValues"]

            # Melt the DataFrame for Seaborn's lineplot
            avg_time_df_melted = pd.melt(avg_time_df, id_vars=["Time"], var_name="Method", value_name="MeanValue")

            # Plotting
            # plt.sca(axes[i])
            g = sns.lineplot(data=avg_time_df_melted, x="Time", y="MeanValue", hue="Method", markers=True,
                             linewidth=2.5, legend=False)
            # palette=method_colors, hue_order=order, linewidth=2.5, legend=False)
            # axes[i].set_title(self.y_label[variable])
            axes.set_xlabel("time [s]", fontsize=12)
            axes.set_ylabel(self.y_label[variable], fontsize=12)
            if variable == "error_x_relative":
                axes.set_yscale("log")

        legend_handles = [
            # Line2D([0], [0], color='tab:green', linewidth=2.5),
            # Line2D([0], [0], color='tab:red', linewidth=2.5),
            # Line2D([0], [0], color='tab:orange', linewidth=2.5),
            Line2D([0], [0], color='tab:blue', linewidth=2.5),
            Line2D([0], [0], color='tab:purple', linewidth=2.5)
        ]

        legend_labels = ['Ours, proposed', "Ours, without pseudo-state", 'Algebraic', "NLS"]
        # legend_labels = ["NLS with good initial guess", "NLS with perfect initial guess"]
        fig.suptitle("Average error evolution of the experiments")
        fig.legend(handles=legend_handles, labels=legend_labels, ncol=2, fontsize=12, loc="upper center",
                   bbox_to_anchor=(0.5, 0.92))
        plt.subplots_adjust(top=0.80, bottom=0.12, left=0.12, right=0.99)

    def boxplot_LOS_comp_time(self, sigma_v=[0.01], sigma_d=[0.1], save_fig=False):
        if self.df is None:
            self.create_panda_dataframe()
        # method_colors = {"NLS": "tab:blue", "losupf": "tab:green", "nodriftupf": "tab:red", "algebraic": "tab:orange", "NLS_p":"tab:purple", "QCQP":"tab:purple"}
        # order = ["losupf", "nodriftupf", "algebraic", "NLS", "QCQP"]
        # order = [ "NLS", "NLS_p"]

        # plt.figure(figsize=(8, 4))
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        # axes_i = 0
        for i, variable in enumerate(["error_x_relative", "error_h_relative"]):  #self.y_label:
            df = self.df.loc[~self.df["Method"].isin(["slam", "WLS", "upf"]) & (self.df["Variable"] == variable) &
                             (self.df["Sigma_uwb"].isin(sigma_d)) & (self.df["Sigma_dv"].isin(sigma_v))]
            methods = df["Method"].unique()

            method_means = []  # to store mean values for each method
            time_points = df["Time"].unique()
            # time_points = time_points[:210]

            for method in methods:
                method_time_values = []  # to store values at each time point for a specific method
                for time_point in time_points:
                    mean_value = df[(df["Method"] == method) & (df["Time"] == time_point)]["value"].mean()
                    method_time_values.append(mean_value)

                method_means.append({"Method": method, "TimeValues": method_time_values})
                print("For Method:", method, variable, "Average over all conditions at each time point:",
                      method_time_values)

            # Create a new DataFrame with average values and time points
            avg_time_df = pd.DataFrame({"Time": time_points})

            for method_mean in method_means:
                avg_time_df[method_mean["Method"]] = method_mean["TimeValues"]

            # Melt the DataFrame for Seaborn's lineplot
            avg_time_df_melted = pd.melt(avg_time_df, id_vars=["Time"], var_name="Method", value_name="MeanValue")

            # Plotting
            plt.sca(axes[i])

            g = sns.lineplot(data=avg_time_df_melted, x="Time", y="MeanValue", hue="Method", markers=True,
                             linewidth=2.5, legend=True)
            # palette=method_colors, hue_order=order,
            axes[i].set_xlabel("time [s]", fontsize=12)
            axes[i].set_ylabel(self.y_label[variable], fontsize=12)
            if variable == "error_x_relative":
                axes[i].set_yscale("log")

        legend_handles = [
            Line2D([0], [0], color='tab:green', linewidth=2.5),
            Line2D([0], [0], color='tab:red', linewidth=2.5),
            Line2D([0], [0], color='tab:orange', linewidth=2.5),
            Line2D([0], [0], color='tab:blue', linewidth=2.5),
            Line2D([0], [0], color='tab:purple', linewidth=2.5),

            # Line2D([0], [0], color='tab:purple', linewidth=2.5)
        ]
        legend_labels = ['Ours, proposed', "Ours, without pseudo-state", 'Algebraic', "NLS", "QCQP"]
        # legend_labels = ["NLS with good initial guess", "NLS with perfect initial guess"]
        # fig.suptitle("Average error evolution of the experiments")
        # fig.legend( handles=legend_handles, labels=legend_labels, ncol=4, fontsize=12, loc="upper center", bbox_to_anchor=(0.5, 0.92))
        # fig.legend( ncol=4, fontsize=12, loc="upper center", bbox_to_anchor=(0.5, 0.92))
        # plt.subplots_adjust(top=0.80, bottom=0.12, left=0.12, right=0.99)

    #-----------------------
    # NLOS functions:
    #-----------------------

    def boxplot_NLOS_comp(self, save_fig=False):
        if self.df is None:
            self.create_panda_dataframe()

        time_len = len(self.df["Time"].unique())

        y_label = {"error_x_relative": "Relative position error [m]",
                   "error_h_relative": "Relative heading error [rad]",
                   "los_error": "LOS error"}
        gs = []
        for variable in y_label:
            df = self.df.loc[
                #(self.df["Time"] == time_len-1) &
                (~self.df["Method"].isin(["slam", "WLS"])) &
                (self.df["Variable"] == variable) &
                (self.df["Sigma_uwb"].isin([1, 0.1])) &
                (self.df["Sigma_dv"].isin([0.1, 0.01]))]
            g = sns.catplot(data=df, kind='box', col='Sigma_uwb', row="Sigma_dv", y='value', x='Method', hue='Method',
                            dodge=False, height=3, aspect=0.65)
            g.tick_params(bottom=False)
            g.set_xticklabels(labels=["", ""])
            g.set_axis_labels("", "")
            g.set_titles("")
            g.fig.suptitle(y_label[variable])
            smallest_x = 0
            smallest_y = 100
            x_values = []
            y_values = []
            for ax in g.axes_dict:
                if ax[0] not in x_values:
                    x_values.append(ax[0])
                if ax[1] not in y_values:
                    y_values.append(ax[1])
                if ax[0] > smallest_x:
                    smallest_x = ax[0]
                if ax[1] < smallest_y:
                    smallest_y = ax[1]

            for sigma_x in x_values:
                if g.axes_dict[(sigma_x, smallest_y)] != None:
                    g.axes_dict[(sigma_x, smallest_y)].set_ylabel("$\\sigma_v $= " + str(sigma_x),
                                                                  fontdict={'fontsize': rcParams['axes.labelsize']})
            for sigma_y in y_values:
                # print(sigma_y, smallest_x)
                if g.axes_dict[(smallest_x, sigma_y)] != None:
                    # g.axes_dict[(smallest_x, sigma_y)].xaxis.set_label_position('top')
                    g.axes_dict[(smallest_x, sigma_y)].set_xlabel("$\\sigma_d $= " + str(sigma_y),
                                                                  fontdict={'fontsize': rcParams['axes.labelsize']})
            # for sigma_d in [0.01, 0.1, 1]:
            #     if g.axes_dict[(sigma_d, 0.01)] != None:
            #         g.axes_dict[(sigma_d, 0.01)].set_ylabel("$\sigma_v $= 0.01")
            # g.axes_dict[(0.01, 0.01)].set_ylabel("$\sigma_v $= 0.01")
            # g.axes_dict[(0.1, 0.01)].set_ylabel("$\sigma_v$ = 0.1")
            # g.axes_dict[(0.01, 0.01)].set_title("$\sigma_d $= 0.1",
            #                                        fontdict={'fontsize': rcParams['axes.labelsize']})
            # g.axes_dict[(0.01, 1)].set_title("$\sigma_d $= 1", fontdict={'fontsize': rcParams['axes.labelsize']})
            if variable == "calculation_time":
                legend_data = g._legend_data
                new_legend_data = {}

                for name in legend_data:
                    new_legend_data[self.names[name]] = legend_data[name]

                g.add_legend(legend_data=new_legend_data)

                for ax in g.axes_dict:
                    g.axes_dict[ax].set_yscale("log")
                #     g.axes_dict[ax].set_ylim(0, 0.4)

            if variable == "error_x_relative":
                for ax in g.axes_dict:
                    g.axes_dict[ax].set_yscale("log")

            g.fig.subplots_adjust(top=0.9)
            g.fig.suptitle(y_label[variable])
            if save_fig:
                g.fig.savefig(variable + ".png")
            gs.append(g)


if __name__ == "__main__":
    result_folder = "../tests/test_cases/RPE_2_agents/Results/test/single_test"

    taa = TwoAgentAnalysis(result_folders=result_folder)
    # taa.delete_data()
    taa.create_panda_dataframe()
    taa.boxplots()
    # taa.boxplot_sigma()
