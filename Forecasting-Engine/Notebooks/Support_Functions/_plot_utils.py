import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def plot_config(title, axis, y_title, x_title):
    """This function is used to configure a plot
    
       Arguments:
            title (str): The main title of the plot
            axis (str): The axis in subplots where the legend will live
            y_title (str): The label for the y-axis
            x_title (str): The label for the x-axis
            
       Returns:
            Sets the configurations for matplotlib plots
    """
    
    # Main Title
    #figure.suptitle('Categorical Plots')

    # Labels
    axis.set_title(title, fontsize=20)
    axis.set_xlabel(x_title, fontsize=15)
    axis.set_ylabel(y_title, fontsize=15)
    
    # Graph Spins
    # Plot 1: Config
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    
    # Tickmark Sizes
    axis.tick_params(axis='both', which='major', labelsize=15)

def create_legend(legend_dict, axis, font_size=15):
    """This function is used to create a legend for matplotlib plots
    
       Arguments:
            legend_dict (dict): A python dictionary with the format {'Label 1':'C0', 'Label 2':'red'}
            axis (plt object): The matplotlib axis where the legend will live
            font_size (int): The font size of the legend, default set to 15
            
       Returns:
            A legend for the specified axis
    """
    list_patches = []
    for label in legend_dict.keys():
        color = legend_dict[label]
        list_patches.append(mpatches.Patch(color=color, label=label))
        axis.legend(handles=list_patches, loc='upper left', prop={'size': font_size})

"""
################################################################################
                               How to use with 1 plot:
################################################################################

# Configure 1 Plot
figure, axis = plt.subplots(figsize = (16,9))
plot_config('Plot Title Here', axis,'','')
create_legend({'Original Data':'C0', 'Other Data':'red'}, axis)

# Plot
plt.plot(data, color='#1f77b4') 

################################################################################
                               How to use with 2+ plot:
################################################################################

# Configure 1 Plot
figure, axis = plt.subplots(1,2, figsize = (16,8))

plot_axis = axis[0]
plot_config('Plot 1 Title Here', plot_axis,'','')
create_legend({'Original Data':'C0', 'Other Data':'red'}, plot_axis)
iter_axis.plot(data, color='#1f77b4') 

plot_axis = axis[1]
plot_config('Plot 2 Title Here', plot_axis,'','')
create_legend({'Original Data':'C0', 'Other Data':'red'}, plot_axis)
sns.histplot(y_hat - y_test, kde = True, ax = plot_axis)

################################################################################
                                 Plot Examples
################################################################################

# Line Plot
plt.plot(data, color='#1f77b4') 

# Hist Plot
sns.histplot(y_hat - y_test, kde = True, ax = plot_axis)

# Scatterplot (Outliers)
plt.scatter(outliers.index, outliers[feature], marker = 'x', color = 'red', s = 250)

# Regression Plot
sns.regplot(y_test, y_hat,lowess=True, line_kws= {'color': 'red', 'linestyle':'--'}, ax=plot_axis)

# Scatter Plot 1
plt.scatter(y_test, y_hat)
"""







