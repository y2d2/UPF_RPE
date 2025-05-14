from matplotlib.lines import Line2D

from Code.Analysis import TwoAgentAnalysis as TAA
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def get_load_methods():
    variables = ["error_x_relative", "error_h_relative"]
    sigma_dv = [0.08]
    sigma_dw = [0.08]
    sigma_uwb = [0.25]

    upf_exp = {"Method": "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
               "Variables": {
                   "Type": ["experiment"],
                   "Variable": variables,
                   "Sigma_dv": sigma_dv,
                   "Sigma_dw": sigma_dw,
                   "Sigma_uwb": sigma_uwb,
                   "Frequency": [10.0],
               },
               "Color": "tab:green",
               "Legend": "Ours",
               }
    upf_exp_per = {"Method": "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0|multi_particles=0",
                   "Variables": {
                       "Type": ["experiment"],
                       "Variable": variables,
                       "Sigma_dv": sigma_dv,
                       "Sigma_dw": sigma_dw,
                       "Sigma_uwb": sigma_uwb,
                       "Frequency": [10.0],
                   },
                   "Color": "tab:orange",
                   "Legend": "Ours *",
                   }
    nodriftupf_exp = {"Method": "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                      "Variables": {
                          "Type": ["experiment"],
                          "Variable": variables,
                          "Sigma_dv": sigma_dv,
                          "Sigma_dw": sigma_dw,
                          "Sigma_uwb": sigma_uwb,
                          "Frequency": [10.0],
                      },
                      "Color": "tab:red",
                      "Legend": r"Ours, $\tilde{\text{w}}$ pseudo-state",
                      }
    alg_exp = {"Method": "algebraic|frequency=10.0|horizon=100",
               "Variables": {
                   "Type": ["experiment"],
                   "Variable": variables,
                   "Sigma_dv": sigma_dv,
                   "Sigma_dw": sigma_dw,
                   "Sigma_uwb": sigma_uwb,
                   "Frequency": [10.0],
               },
               "Color": "tab:brown",
               "Legend": "Algebraic",
               }
    qcqp_exp = {"Method": "QCQP|frequency=10.0|horizon=100",
                "Variables": {
                    "Type": ["experiment"],
                    "Variable": variables,
                    "Sigma_dv": sigma_dv,
                    "Sigma_dw": sigma_dw,
                    "Sigma_uwb": sigma_uwb,
                    "Frequency": [10.0],
                },
                "Color": "tab:blue",
                "Legend": "QCQP",
                }
    nls_exp = {
        "Method": "NLS|frequency=1.0|horizon=10",
        "Variables": {
            "Type": ["experiment"],
            "Variable": variables,
            "Sigma_dv": sigma_dv,
            "Sigma_dw": sigma_dw,
            "Sigma_uwb": sigma_uwb,
            "Frequency": [1.0],
        },
        "Color": "tab:purple",
        "Legend": "NLS *",
    }

    upf_sim = {"Method": "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
               "Variables": {
                   "Type": ["simulation"],
                   "Variable": variables,
                   "Sigma_dv": sigma_dv,
                   "Sigma_dw": sigma_dw,
                   "Sigma_uwb": sigma_uwb,
                   "Frequency": [10.0],
               },
               "Color": "lightgreen",
               "Legend": "Ours (sim)",
               }
    upf_sim_per = {"Method": "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0|multi_particles=0",
                   "Variables": {
                       "Type": ["simulation"],
                       "Variable": variables,
                       "Sigma_dv": sigma_dv,
                       "Sigma_dw": sigma_dw,
                       "Sigma_uwb": sigma_uwb,
                       "Frequency": [10.0],
                   },
                   "Color": "bisque",
                   "Legend": "Ours * (sim)",
                   }
    nodriftupf_sim = {"Method": "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                      "Variables": {
                          "Type": ["simulation"],
                          "Variable": variables,
                          "Sigma_dv": sigma_dv,
                          "Sigma_dw": sigma_dw,
                          "Sigma_uwb": sigma_uwb,
                          "Frequency": [10.0],
                      },
                      "Color": "salmon",
                      "Legend": r"Ours, $\tilde{\text{w}}$ pseudo-state (sim)",
                      }
    alg_sim = {"Method": "algebraic|frequency=10.0|horizon=100",
               "Variables": {
                   "Type": ["simulation"],
                   "Variable": variables,
                   "Sigma_dv": sigma_dv,
                   "Sigma_dw": sigma_dw,
                   "Sigma_uwb": sigma_uwb,
                   "Frequency": [10.0],
               },
               "Color": "chocolate",
               "Legend": "Algebraic (sim)",
               }
    qcqp_sim = {"Method": "QCQP|frequency=10.0|horizon=100",
                "Variables": {
                    "Type": ["simulation"],
                    "Variable": variables,
                    "Sigma_dv": sigma_dv,
                    "Sigma_dw": sigma_dw,
                    "Sigma_uwb": sigma_uwb,
                    "Frequency": [10.0],
                },
                "Color": "cornflowerblue",
                "Legend": "QCQP (sim)",
                }
    nls_sim = {
        "Method": "NLS|frequency=1.0|horizon=10",
        "Variables": {
            "Type": ["simulation"],
            "Variable": variables,
            "Sigma_dv": sigma_dv,
            "Sigma_dw": sigma_dw,
            "Sigma_uwb": sigma_uwb,
            "Frequency": [1.0],
        },
        "Color": "thistle",
        "Legend": "NLS (sim)",
    }

    methods_order_sim = [upf_exp, upf_sim,

                         # nodriftupf_exp, nodriftupf_sim,
                         # alg_exp, alg_sim,
                         qcqp_exp, qcqp_sim,
                         upf_exp_per, upf_sim_per,
                         nls_exp, nls_sim
                         ]
    return methods_order_sim

def boxplots(df, methods_names, methods_colors, methods_legends):
    g = taa.boxplot_exp(df, methods_color=methods_colors, methods_legend=methods_legends,
                        hue_variable="Name", hue_order=methods_names,
                        col_variable="Variable",
                        row_variable=None,
                        x_variable="Sigma_dv",
                        )

    # g.axes_dict["error_x_relative"].set_yscale("log")
    g.axes_dict["error_h_relative"].set_ylabel(taa.y_label["error_h_relative"])
    g.axes_dict["error_x_relative"].set_ylabel(taa.y_label["error_x_relative"])
    sns.move_legend(g, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=4)
    plt.tight_layout()
    plt.subplots_adjust(top=0.8, bottom=0.12, left=0.06, right=0.99)

    g.savefig("../../../fig/boxplot.png", dpi=600, pad_inches=0.3)
    # g.set_dpi(600)
    # g.savefig("../../../fig/boxplot.png", dpi=600)


def lineplots(df, methods_names, methods_colors, methods_legends):
    g = taa.lineplot(df, methods_names, methods_colors,
                     methods_legends=methods_legends)
    g[0].set_ylabel(taa.y_label["error_x_relative"])
    g[1].set_ylabel(taa.y_label["error_h_relative"])
    g[0].set_xlabel("time [s]")
    g[1].set_xlabel("time [s]")
    legend_handles = [Line2D([0], [0], color=methods_colors[method], linewidth=2.5) for method in methods_names]
    legend_labels = [methods_legends[method] for method in methods_names]
    # fig.suptitle("Average error evolution of the experiments")
    fig = plt.gcf()
    fig.legend(handles=legend_handles, labels=legend_labels, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 0.98))
    plt.tight_layout()
    plt.subplots_adjust(top=0.75, bottom=0.12, left=0.06, right=0.99)

    plt.savefig("../../../fig/lineplot.png", dpi=600, pad_inches=0.3)
    # g.axes_dict["error_h_relative"].set_ylabel(taa.y_label["error_h_relative"])
    # g.axes_dict["error_x_relative"].set_ylabel(taa.y_label["error_x_relative"])

#
result_folders = [
    "../../../Results/experiments",
    # "../../../Results/sim2real",
]
taa = TAA.TwoAgentAnalysis(result_folders=result_folders)
methods_order = get_load_methods()
df, methods_names, methods_colors, methods_legends = taa.filter_methods_new(methods_order)


taa.print_statistics(methods_names, ["error_x_relative", "error_h_relative"], df)
sns.set_context("talk", font_scale=1.8)  # "talk" or "poster" context
sns.set_style("whitegrid")

# boxplots(df, methods_names, methods_colors, methods_legends)



methods_names = [
    # "algebraic|frequency=10.0|horizon=100|Type_['experiment']|Variable_['error_x_relative', 'error_h_relative']|Sigma_dv_[0.08]|Sigma_dw_[0.08]|Sigma_uwb_[0.25]|Frequency_[10.0]",
    "QCQP|frequency=10.0|horizon=100|Type_['experiment']|Variable_['error_x_relative', 'error_h_relative']|Sigma_dv_[0.08]|Sigma_dw_[0.08]|Sigma_uwb_[0.25]|Frequency_[10.0]",
    "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0|multi_particles=0|Type_['experiment']|Variable_['error_x_relative', 'error_h_relative']|Sigma_dv_[0.08]|Sigma_dw_[0.08]|Sigma_uwb_[0.25]|Frequency_[10.0]",
    # "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0|Type_['experiment']|Variable_['error_x_relative', 'error_h_relative']|Sigma_dv_[0.08]|Sigma_dw_[0.08]|Sigma_uwb_[0.25]|Frequency_[10.0]",
    "NLS|frequency=1.0|horizon=10|Type_['experiment']|Variable_['error_x_relative', 'error_h_relative']|Sigma_dv_[0.08]|Sigma_dw_[0.08]|Sigma_uwb_[0.25]|Frequency_[1.0]",
    "losupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0|Type_['experiment']|Variable_['error_x_relative', 'error_h_relative']|Sigma_dv_[0.08]|Sigma_dw_[0.08]|Sigma_uwb_[0.25]|Frequency_[10.0]",
]

methods_colors["QCQP|frequency=10.0|horizon=100|Type_['experiment']|Variable_['error_x_relative', 'error_h_relative']|Sigma_dv_[0.08]|Sigma_dw_[0.08]|Sigma_uwb_[0.25]|Frequency_[10.0]"] = "cornflowerblue"

lineplots(df, methods_names, methods_colors, methods_legends)



#
plt.show()