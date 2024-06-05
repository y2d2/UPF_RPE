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
    def __init__(self, result_folder):
        self.result_folder = result_folder
        self.results = None
        self.data = None
        self.dfs = []
        self.df = None
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
        n_files = len(os.listdir(self.result_folder))
        file_nr = 0.
        for file in os.listdir(self.result_folder):
            file_nr +=1
            if file.endswith(".pkl"):
                with open(self.result_folder + "/" + file, "rb") as f:
                    try:
                        data = pkl.load(f)
                        print("Loading "+ str(int(file_nr/n_files*100.)), "%: "+  self.result_folder + "/" + file)
                    except EOFError: print("!!!!!!!!! Could not open: ", self.result_folder + "/" + file + " !!!!!!!!!")
                f.close()
                if "numerical_data" not in data:
                    print("Reformating the data for analysis " + file + " ...")
                    data = self.reformat_data(data)
                    with open(self.result_folder + "/" + file, "wb") as f:
                        pkl.dump(data, f)
                    f.close()
                self.data[file] = data

                if "analysis" not in data:
                    print("Starting statistical analysis of " + file + " ...")
                    data = self.calculate_statistics(data)
                    with open(self.result_folder + "/" + file, "wb") as f:
                        pkl.dump(data, f)
                    f.close()
                self.results[file] = data["analysis"]

                if "panda_date" not in data:
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
                                              Frequency=data["parameters"]["frequency"])
                self.dfs.append(df)

            # df_var = pd.concat([dfs[method][variable] for variable in dfs[method]])
            # df_var = pd.melt(df_var, id_vars=['Variable', 'Method', 'Sigma_dv', 'Sigma_uwb'], var_name=["Time [s]"])  # MELT
            # dfs[method] = df_var

    def create_panda_dataframe(self):
        if self.data is None:
            self.load_results()
        dfs = pd.concat(self.dfs)
        self.df = pd.melt(dfs, id_vars=['Variable', 'Method', 'Sigma_dv', 'Sigma_dw','Sigma_uwb', "Frequency"],
                          var_name=["Time"])  # MELT
        return

    # -----------------------
    # delete functions:
    # -----------------------
    def delete_data(self):
        self.delete_statistical_analysis()
        self.delete_numerical_data()

    def delete_statistical_analysis(self):
        if self.data is None:
            self.load_results()

        for file in self.data:
            if "analysis" in self.data[file]:
                del self.data[file]["analysis"]
                with open(self.result_folder + "/" + file, "wb") as f:
                    pkl.dump(self.data[file], f)
                f.close()

    def delete_numerical_data(self):
        if self.data is None:
            self.load_results()

        for file in self.data:
            if "numerical_data" in self.data[file]:
                del self.data[file]["numerical_data"]
                with open(self.result_folder + "/" + file, "wb") as f:
                    pkl.dump(self.data[file], f)
                f.close()

    # -----------------------
    # Support plotting functions:
    # -----------------------
    def remove_x_ticks(self, g, frequencies):
        # g.tick_params(bottom=False)
        g.set_axis_labels()

        if len(frequencies) > 1:
            g.set_xticklabels(labels=[str(int(frequency)) + "Hz" for frequency in frequencies])
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


    def set_legend(self, g, methods_legend = {}):
        if methods_legend == {}:
            g.add_legend()
            return

        legend_data = g._legend_data
        new_legend_data = {}

        print(legend_data)
        for name in methods_legend:
            new_legend_data[methods_legend[name]] = legend_data[name]

        g.add_legend(legend_data=new_legend_data)


    def print_statistics(self, methods, variable, df):
        for method in methods:
            print(method, variable, df[df["Method"] == method]["value"].mean(), " pm ",
                  df[df["Method"] == method]["value"].std(), "; median: ",
                  df[df["Method"] == method]["value"].median())

    def filter_methods(self, methods, sigma_uwb, sigma_v, frequencies, start_time_index):
        if self.df is None:
            self.create_panda_dataframe()

        if len(methods) == 0:
            methods = self.df["Method"].unique()

        df = self.df.loc[(self.df["Method"].isin(methods)) &
                         (self.df["Sigma_uwb"].isin(sigma_uwb)) &
                         (self.df["Sigma_dv"].isin(sigma_v)) &
                         (self.df["Time"] > start_time_index) &
                         (self.df["Frequency"].isin(frequencies))
                         ]
        return df, methods

    # -----------------------
    # Plotting functions:
    # -----------------------
    def boxplot_var(self, number_of_bins=5, sigma_uwb=0.1, sigma_dv=0.1):
        if self.df is None:
            self.create_panda_dataframe()

        time_len = len(self.df["Time"].unique())
        time_bins = np.linspace(0, time_len - 1, number_of_bins).astype(int)

        gs = []
        for variable in self.y_label:
            df = self.df.loc[(self.df["Time"].isin(time_bins)) &
                             (self.df["Method"] != "slam") &
                             (self.df["Variable"] == variable)]
            g = sns.catplot(data=df, kind='box', col='Time', y='value', x='Method', hue='Method', dodge=False,
                            height=3, aspect=0.65)
            g.tick_params(bottom=False)
            g.set_xticklabels(labels=["", "", "", ""])
            g.set_axis_labels("", self.y_label[variable])
            gs.append(g)

        legend_data = gs[0]._legend_data
        new_legend_data = {}

        for name in legend_data:
            new_legend_data[self.names[name]] = legend_data[name]

        gs[0].add_legend(legend_data=new_legend_data)
        for i in range(1, len(gs)):
            gs[i].set_titles("")



    def boxplot_LOS_comp(self, sigma_uwb=[1., 0.1], sigma_v=[0.1, 0.01],  frequencies = [1.0, 10.0],
                         methods_order=[], start_time_index=0, methods_color=None, methods_legend = {},  save_fig=False):
        method_df, methods_order = self.filter_methods(methods_order, sigma_uwb, sigma_v, frequencies, start_time_index)
        gs = []

        for variable in self.y_label:
            df = method_df.loc[(self.df["Variable"] == variable)]

            # if methods_color == {}:
            #     g = sns.catplot(data=df, kind='box', col='Sigma_uwb', row="Sigma_dv", y='value', x='Method', hue='Method',
            #                 dodge=False, height=3, aspect=0.65, order=methods_order)
            # else:
            # g = sns.catplot(data=df, kind='box', col='Sigma_uwb', row="Sigma_dv", y='value', x='Uwb_rate', hue='Method',
            #             dodge=False, height=3, aspect=0.65, order=methods_order, palette=methods_color, hue_order=methods_order)
            g = sns.catplot(data=df, kind='box', col='Sigma_uwb', row="Sigma_dv", y='value', x='Frequency', hue='Method',
                                    dodge=True, aspect=0.65, order= frequencies, palette=methods_color, hue_order=methods_order,
                            legend=False)

            self.remove_x_ticks(g, frequencies)
            self.set_labels(g)
            self.print_statistics(methods_order, variable, df)



            if variable == "calculation_time":
                self.set_legend(g, methods_legend)
                for ax in g.axes_dict:
                    g.axes_dict[ax].set_yscale("log")

            if variable == "error_x_relative":
                for ax in g.axes_dict:
                    g.axes_dict[ax].set_yscale("log")

            g.fig.subplots_adjust(top=0.9)
            g.fig.suptitle(self.y_label[variable])
            if save_fig:
                g.fig.savefig(self.result_folder + "/" + variable + ".png")
            gs.append(g)

    def box_plot(self, methods=None, save_fig=False):
        if self.df is None:
            self.create_panda_dataframe()

        if methods is None:
            methods = self.df["Method"].unique()
        gs = []
        for variable in self.y_label:
            df = self.df.loc[(self.df["Sigma_uwb"].isin([0.1])) &
                             (self.df["Sigma_dv"].isin([0.01])) &
                             (self.df["Method"].isin(methods)) &
                             (self.df["Variable"] == variable)]
            methods = df["Method"].unique()
            print(methods)

    def boxplot_freq_comp(self, save_fig=False):

        if self.df is None:
            self.create_panda_dataframe()

        gs = []
        for variable in self.y_label:
            df = self.df.loc[(self.df["Sigma_uwb"].isin([0.1])) &
                             (self.df["Sigma_dv"].isin([0.01])) &
                             (~self.df["Method"].isin(["slam", "WLS", "upf"])) &
                             (self.df["Variable"] == variable)]
            methods = df["Method"].unique()
            print(methods)
            # if variable=="calculation_time":
            #     df_losupf = df.loc[df["Method"] == "losupf"]
            #     df_losupf['Method'] =  df_losupf['Method'].replace(['losupf'], 'nodriftupf')
            #
            #     df = df.loc[(~self.df["Method"].isin(["nodriftupf"]))]
            #     print(df["Method"].unique())
            #     df= pd.concat([df, df_losupf])
            #     print(df["Method"].unique())

            # df.append(df_nodriftupf)
            # df.loc[df["Method"] == "nodriftupf"] = df.loc[df["Method"] == "losupf"]
            # df[df["Method"] == "nodriftupf"] = df[df["Method"] == "losupf"]
            for method in methods:
                print(method, variable, df[df["Method"] == method]["value"].mean(), " pm ",
                      df[df["Method"] == method]["value"].std(), "; median: ",
                      df[df["Method"] == method]["value"].median())
            # df.loc[df["Method"] ==
            # # print(df.mean(), df.std())

            # order = ["losupf|resample_factor=0.1|sigma_uwb_factor=2.0"]
            g = sns.catplot(data=df, kind='box', col='Uwb_rate', row="Sigma_uwb", y='value', x='Method', hue='Method',
                            dodge=False, height=3, aspect=0.65)

            g.tick_params(bottom=False)
            g.set_axis_labels()
            ticks_nr = len(set(df["Method"].unique()) - {"slam", "WLS", "upf"})
            g.set_xticklabels(labels=["" for i in range(ticks_nr)])
            g.set_axis_labels("", "")
            g.set_titles("")

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

            # for sigma_x in x_values:
            #     if g.axes_dict[(sigma_x, smallest_y)] != None:
            #         g.axes_dict[(sigma_x, smallest_y)].set_ylabel("VIO std: $\\sigma_v $= " + str(sigma_x), fontdict={'fontsize': rcParams['axes.labelsize']})
            # for sigma_y in y_values:
            #     if g.axes_dict[(smallest_x, sigma_y)] != None:
            #         g.axes_dict[(smallest_x, sigma_y)].set_xlabel("UWB std: $\\sigma_d $= " + str(sigma_y), fontdict={'fontsize': rcParams['axes.labelsize']})

            if variable == "calculation_time":
                g.add_legend()
            #     legend_data = g._legend_data
            #     new_legend_data = {}
            #
            #     print(legend_data)
            #     for name in order:
            #         new_legend_data[self.names[name]] = legend_data[name]
            #
            #     g.add_legend(legend_data=new_legend_data)
            #
            #     for ax in g.axes_dict:
            #         g.axes_dict[ax].set_yscale("log")

            if variable == "error_x_relative":
                for ax in g.axes_dict:
                    g.axes_dict[ax].set_yscale("log")

            g.fig.subplots_adjust(top=0.9)
            g.fig.suptitle(self.y_label[variable])
            if save_fig:
                g.fig.savefig(self.result_folder + "/" + variable + ".png")
            gs.append(g)

    def single_settings_boxplot(self, save_fig=False):
        if self.df is None:
            self.create_panda_dataframe()

        order = ["losupf", "nodriftupf", "algebraic", "NLS", "QCQP"]

        # Filter the DataFrame for the desired variables and methods
        df = self.df.loc[
            (~self.df["Method"].isin(["slam", "WLS", "upf"])) &
            (self.df["Variable"].isin(["error_x_relative", "error_h_relative"]))
            ]

        # Create custom subplots with different y-axis scales
        fig, axes = plt.subplots(1, 2, figsize=(7, 4))
        fig.suptitle(
            r"Experiments: $\sigma_v= 0.15 \frac{m}{s}$, $\sigma_w=0.05 \frac{(rad)}{s}$ and $\sigma_d = 0.25 m$",
            fontsize=12)
        for i, variable in enumerate(["error_x_relative", "error_h_relative"]):
            df_c = df[df["Variable"] == variable]
            methods = df_c["Method"].unique()
            for method in methods:
                print(method, variable, df_c[df_c["Method"] == method]["value"].mean(), " pm ",
                      df_c[df_c["Method"] == method]["value"].std(), "; median: ",
                      df_c[df_c["Method"] == method]["value"].median())

            order = ["losupf", "nodriftupf", "algebraic", "NLS"]
            custom_colors = {"losupf": "tab:green", "nodriftupf": "tab:red", "algebraic": "tab:orange",
                             "NLS": "tab:blue"}
            g = sns.boxplot(data=df[df["Variable"] == variable], x='Method', y='value', hue='Method', dodge=False,
                            ax=axes[i], order=order, palette=custom_colors)

            axes[i].set_title("")
            axes[i].set_ylabel("")
            # axes[i].set_xlabel("Method")
            axes[i].set_xlabel(self.y_label[variable], fontsize=12)
            axes[i].tick_params(bottom=False)
            axes[i].set_xticklabels([])
            if variable == "error_x_relative":
                axes[i].set_yscale("log")
            else:
                axes[i].set_yscale("linear")
            # legend_data = g._legend_data
        # Remove the legend from the first subplot
        axes[0].get_legend().remove()
        # axes[1].get_legend().remove()

        # Customize the legend for the second subplot and move it outside to the right
        legend_data = axes[1].get_legend().legendHandles

        new_legend_data = {}

        for line, text in zip(legend_data, df["Method"].unique()):
            new_legend_data[self.names[text]] = line
        legend_data = new_legend_data
        new_legend_data = {}
        for name in order:
            new_legend_data[self.names[name]] = legend_data[self.names[name]]
        # Position the legend to the right outside of the second subplot
        # axes[1].legend(new_legend_data.values(), new_legend_data.keys(), loc='upper left', bbox_to_anchor=(1.05, 0.75))

        # Adjust the spacing between subplots to create more room for the legend
        plt.subplots_adjust(wspace=0.5, right=0.7)  # Adjust the 'wspace' value as needed
        # You can save the figure to a file if needed

        # fig.suptitle("Average error over all simulated conditions")
        fig.legend(handles=new_legend_data.values(), labels=new_legend_data.keys(), ncol=4, fontsize=10,
                   loc="upper center", bbox_to_anchor=(0.5, 0.92))
        axes[1].get_legend().remove()
        plt.subplots_adjust(top=0.80, bottom=0.12, left=0.07, right=0.93)

        if save_fig:
            plt.savefig(self.result_folder + "/" + "single_settings_boxplot.pdf", bbox_inches='tight')

    #------------------------
    # Time analysis
    #------------------------
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

    taa = TwoAgentAnalysis(result_folder=result_folder)
    # taa.delete_data()
    taa.create_panda_dataframe()
    taa.boxplot_var()
    # taa.boxplot_sigma()
