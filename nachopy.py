import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import copy

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def basic_eda(df, df_name = 'dataframe', showc = True):
    """
    getting some basic information about each dataframe
    shape of dataframe i.e. number of rows and columns
    total number of rows with null values
    total number of duplicates
    data types of columns
    Args:
        df (dataframe): dataframe containing the data for analysis
        df_name (string): name of the dataframe
    """
    print(df_name.upper())
    print()
    print(f"Rows: {df.shape[0]} \t Columns: {df.shape[1]}")
    print()
    print(f"Total null rows: {df.isnull().sum().sum()}")
    print(f"Percentage null rows: {round(df.isnull().sum().sum() / df.shape[0] * 100, 2)}%")
    print()
    print(f"Total duplicate rows: {df[df.duplicated(keep=False)].shape[0]}")
    print(f"Percentage dupe rows: {round(df[df.duplicated(keep=False)].shape[0] / df.shape[0] * 100, 2)}%")
    print()
    print(df.info(show_counts = showc))
    
    
def digit_checker(df, d_min = 0, d_max = 6):
    for i in range(d_min,d_max):
        d_no = 10**i

        validndc_bool = df.ndc>=d_no
        print(f'% of NDCs >= 10^{i}:', round(validndc_bool.sum()/len(df)*100,2))
        print('Number of NDCs:', df.ndc[validndc_bool].nunique(),'\n')
    
    
    

def cat_check (df,cat = 'A', ratio = True, cl = -1): 
    '''
    Simple function to check distribution of categories. By default returns
    % of missing values (cat = A and class(cl) = -1)
    '''
    if df[cat].min() >= 0:
        return 0
    else:
        return df[cat].value_counts(normalize = ratio).loc[cl]
    
    
def cat_filler(df,filler_col,cat = list('ABCDGHJLMNPRSV')):
    # Modifies dataframe (inplace always True)!! Create copy beforehand
    # Columns of grouped dataframe
    catcol = copy.deepcopy(cat)
    catcol.append(filler_col)
    # Getting the maximum value of columns grouped by filler (categories are pooled for drugs with same category)
    # This assumes that missing values are labelled as -1
    cat_filler = df[catcol].groupby(filler_col).max()
    # Only interested in the non-missing values
    cat_filler = cat_filler[cat_filler>=0]

    # Setting the index as the filler column for update to work
    df.set_index(filler_col,inplace = True, drop = False)
    # Updating the dataframe only on negative values 
    df.update(cat_filler, filter_func = lambda x: x == -1, overwrite = False)
    # Setting categorical columns as int8 again
    catcol.pop()
    df[catcol] = df[catcol].astype('int8')
    
    df.reset_index(drop = True, inplace = True)
    
    
def rec_cat_filler(df, filler_cols, cat = list('ABCDGHJLMNPRSV')):
    '''    
    Modifies dataframe to fill dummy variables (inplace always True)!! Create copy beforehand
    '''
    # Column exploded upon first element of filler_cols
    new_missing_cats = cat_check(df)*100
    print('% of missing values before loop:', new_missing_cats)
    # Function is called several times. If gsn already split, values in gsn column
    # will be a list and splitting will not be attempted again (otherwise results is null column)
    if isinstance(df.gsn.loc[df.gsn.first_valid_index()],str):
        df[filler_cols[0]] = df[filler_cols[0]].str.split()
    print(f'{filler_cols[0]} column split, starting loop...')
    
    missing_cats = 100
    i = 1
    catcols = copy.deepcopy(cat)
    while new_missing_cats < missing_cats:
        missing_cats = copy.copy(new_missing_cats)

        exploded = df.explode(filler_cols[0]) 
        exploded.reset_index(drop = False, inplace = True)
        print(f'Dataframe exploded, starting for loop #{i}\n')
        for col in filler_cols:
            cat_filler(exploded,col,catcols)
            new_missing_cats = cat_check(exploded)*100

            print(f'Loop #{i}, missing values % after filling with {col} column:',new_missing_cats)

            if new_missing_cats == 0:
                break

        exploded = exploded.set_index('index')[catcols].groupby('index').max()
        df.update(exploded,filter_func=lambda x: x == -1, overwrite = False) # Updating into main df
        new_missing_cats = cat_check(df)*100

        print(f'\nLoop #{i}, missing values % after re-imploding df:',new_missing_cats)

        if new_missing_cats == 0:
            break

        i += 1    
    

def display_sbs(dfs_list, max_rows = 200, suffix = 'table', titles = ['']):
    
    '''
    Displays dataframes as html tables and side by side (if they do not fit, they are
    represented below)
    '''
    from IPython.display import display_html
    
    # If titles list empty or not all titles given, empy titles will be assigned

    html_tables = ''
    for i, df in enumerate(dfs_list): # Iterating over all the dfs added as args
        
        if isinstance(df,pd.core.series.Series): # If df is acc series, convert to df
            df = df.to_frame()
        
        # If titles undefined or not enough titles
        if titles == [''] or i >= len(titles):
            # title equal to the string-sum of column names with commas (removing last comma with [-2]) + _suffix
            title = (df.columns.values + ', ').sum()[:-2] + ' ' + suffix
        else:
            title = titles[i]
        
        first_row = 0
        last_row = max_rows
        prints_no = int(np.ceil(len(df)/max_rows)) if len(df) >= max_rows else 1
        
        for _ in range(prints_no):
            df_s = df.iloc[first_row:last_row]
            # First converted to style, adding caption 
            df_s = df_s.style.set_table_attributes("style='display:inline'").set_caption(title)
            # Style can be converted to html
            html_tables += df_s.to_html()

            first_row = copy.copy(last_row)
            last_row += max_rows
            title = '' # After first loop title is empty
            
    # And displayed using display_html
    display_html(html_tables,raw = True)
    
    pass 



def change_all(X_tr, y_tr, X_ts, y_ts, condition, feature = 'target', change = 'drop', modify = True):
     
    if feature == 'target':
        ind_tr = y_tr.index[condition(y_tr)]
        ind_ts = y_ts.index[condition(y_ts)]
    else:
        ind_tr = X_tr.index[condition(X_tr[feature])]
        ind_ts = X_ts.index[condition(X_ts[feature])]
    
    if modify:

        if change == 'drop':
            X_tr.drop(ind_tr, inplace = True)
            y_tr.drop(ind_tr, inplace = True)
            X_ts.drop(ind_ts, inplace = True)
            y_ts.drop(ind_ts, inplace = True)

        else:
            X_tr.loc[ind_tr, feature] = change
            X_ts.loc[ind_ts, feature] = change
        
    else:
        
        if change == 'drop':
            X_tr_mod = X_tr.drop(ind_tr)
            y_tr_mod = y_tr.drop(ind_tr)
            X_ts_mod = X_ts.drop(ind_ts)
            y_ts_mod = y_ts.drop(ind_ts)

        else:
            X_tr_mod = X_tr.copy()
            X_ts_mod = X_ts.copy()
            
            X_tr_mod.loc[ind_tr, feature] = change
            X_ts_mod.loc[ind_ts, feature] = change
         
        return X_tr_mod, X_ts_mod, y_tr_mod, y_ts_mod



def roc_n_confusion(fittedgrid, X_ts, y_ts, titles = ['',''], normalize = 'true',plots = True,ret = False,lbl_enc = None,incl_val = True):
    '''
    This function takes a grid and prints and returns best cv and test scores.
    Also lots the ROC and the confusion matrix (using RocCurveDisplay and ConfusionMatrixDisplay 
    from sklearn) if plots = True. Titles can be passed as list of two strings.
    '''

    # Imports
    from sklearn.metrics import RocCurveDisplay
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import ConfusionMatrixDisplay

    # Scoring from provided dataset
  
    test_score = fittedgrid.score(X_ts, y_ts)
    if isinstance(fittedgrid,GridSearchCV):
        print("Best model's CV score:",fittedgrid.best_score_)
    print("Best model's test score",test_score)
    
    if len(np.unique(y_ts)) > 2: # if y_test non binary (multiclass problem)
        
        maj_class_prob = np.unique(y_ts,return_counts=True)[1].max()/len(y_ts)
        print("Baseline model (score of a model that always predicts majoritary class)",round(maj_class_prob,2))
        
        if plots:
            from sklearn.preprocessing import LabelBinarizer
            
            # First plot is confusion matrix, using a fitted grid and the datasets
            
            plt.figure(figsize = (4,4))
             
            sns.set(font_scale = 0.6)
            
            labels = lbl_enc.inverse_transform(np.unique(y_ts))
            ConfusionMatrixDisplay.from_estimator(fittedgrid, X_ts, y_ts, normalize = normalize,display_labels = labels,include_values = incl_val,values_format = '.1f')
            
            sns.set(font_scale = 1)

            plt.yticks(rotation = 0,fontsize = 8)
            plt.xticks(rotation = 90,fontsize = 8)
            plt.grid(False)

            plt.figure(figsize = (4,4))
            # Second plot is the roc
            
            y_ts_ohe = LabelBinarizer().fit_transform(y_ts)
            y_probs = fittedgrid.predict_proba(X_ts)
            
            
            RocCurveDisplay.from_predictions(y_ts_ohe.ravel(), y_probs.ravel())
            
            plt.title(titles[1],pad = 10)
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            
            plt.show()

        if ret:
            return (fittedgrid.best_score_, test_score)
    
    
    else: # Binary classification
        
        print("Baseline model (score of a model that always predicts 1)",y_ts.mean())

        if plots:
            

            plt.subplots(1,2)

            # First plot is confusion matrix, using a fitted grid and the datasets
            ax_conf = plt.subplot(1,2,1)
            ConfusionMatrixDisplay.from_estimator(fittedgrid, X_ts, y_ts, ax = ax_conf, normalize = 'true')
            plt.grid(False)
            plt.title(titles[0],pad = 10)

            # Second plot is the roc
            ax_roc = plt.subplot(1,2,2)
            
            plt.title(titles[1],pad = 10)
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')

            plt.show()

        if ret:
            return (fittedgrid.best_score_, test_score)

    
    
    
def label_eval(df, col, title = '', ylab = '',  y = 'Reviewer_Score'):
    
    '''
    Function to evaluate labelling of categorical variables based on distribution and relationsip
    with output class. Takes df, the column to evaluate (col) and y, along with titles.
    '''
    
    # Relationship with the output class (y)
    colvsy = (df[[col,y]].groupby(col).\
        mean()*100).sort_values(by = y, ascending = False)

    # Joining with the value counts so that they can be sorted with
    # based on relationship with y
    colvsy = colvsy.join(df[col].value_counts(normalize = True)*100)
    colvsy.rename(columns = {col: 'Counts'}, inplace = True)

    fig,axes = plt.subplots(1,2)
    fig.suptitle(title,y = 1)

    # Left subplot is value of y per value of evaluated column
    plt.subplot(1,2,1)
    colvsy.Reviewer_Score.plot.barh(legend = False, ax = axes[0])
    plt.ylabel(ylab)
    plt.xlabel('Good reviews (%)')

    # Right plot is value counts of evaluated column
    plt.subplot(1,2,2)
    colvsy.Counts.plot.barh(legend = False, ax = axes[1], color = 'blue')
    plt.xlabel('Review count (%)')
    plt.ylabel('')
    plt.tick_params(axis='y', left=False, labelleft=False)

    plt.tight_layout()
    plt.show()
    
    
def corr_p(df,y_col,fig_title = '',ret = False,fsize = (6,8)):
    
    '''
    Objects passed are a dataframe and one or two strings. From the dataframe
    a 'x' dataframe and a 'y' dataframe are obtained from the numerical
    columns of df and based on column name specified on y_col.
    Calculates the Pearson correlation coefficient (r) along with p-values 
    using stats.pearsonr and plots them as heatmaps if a figure title
    is specified. By default, heatmaps are not plotted. 

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame from which to extract x and y and calculate the
        correlation coefficient. Categoricla columns will be ignored.

    y_col : string
        Determines column used as dependent variable. Column it addresses
        must be numerical. 

    fig_title : string
        Title of subplots figure. If empty, no plots are calculated

    Returns
    -------
    x: Series or DataFrame
        Series of DataFrame with dependent variable(s) (all numerical
        columns from df except for column y_col)

    y: Series
        Independent variable (column in df determined by y_col)

    pearson_r: DataFrame
        Includes the Pearson correlation coefficients (r) of dependent variables 
        (x) with independent variable (y) and the p-values which indicate
        statistical significance of r. For more info, check documentation for
        scipy.stats.pearsonr.

    Notes
    -----
    For columns with constant values, the correlation coefficient is not defined 

    Examples
    --------
    >>> df = pd.DataFrame([[4, 9]] * 3, columns=['A', 'B'])
    >>> df
       A  B
    0  4  9
    1  5  2
    2  4  7

    >>> corr_p(df,'B')
    (   
        A
     0  4
     1  5
     2  4,

          B
     0    9
     1    2
     2    7

               r   p-value
     A -0.960769  0.178912
     )
    '''
    from scipy import stats

    
    # Only numerical columns used
    cols_num = list(df.select_dtypes('number').columns) # List of columns of dtype number
    num_df = df[cols_num].copy() 
    
    x = num_df.drop(columns = y_col) # x contains the dependent variables
    y = num_df[y_col].copy() # y is the dependent variable
    
    # Pearson correlation coefficient applied columnwise 
    pearson_r = x.apply(lambda col: stats.pearsonr(col,y))
    # Renaming output dataframe
    pearson_r = (pearson_r.rename({0:'r',1:'p-value'})).T  

    if fig_title != '': # If third argument specified (as anything other than empty string)
        fig,_ = plt.subplots(1,2,figsize = fsize)
        
        # Defining minmax to select symmetrical limits for the colormap.
        minmax = round(max([abs(min(pearson_r['r'])),abs(max(pearson_r['r']))])+0.01,1)
        ax_r = plt.subplot(1,2,1)
        # Heatmap of the output r column sorted by absolute value of r
        sns.heatmap(pearson_r.sort_values(by = 'r', key = abs, ascending = False)[['r']],
                    cmap = 'coolwarm', 
                    ax = ax_r, 
                    vmin = -minmax, 
                    vmax = minmax,
                    annot = True,
                    xticklabels = False,
                    cbar_kws = {'aspect':round(len(pearson_r)*(3/4)),'pad': 0.01}
                    )
        plt.title('Pearson correlation coefficients of\nfeatures with dependent variable', fontsize = 11)

        # Heatmat of p-values
        ax_p = plt.subplot(1,2,2)
        sns.heatmap(pearson_r.sort_values(by = 'r', key = abs, ascending = False)[['p-value']], 
                    cmap = 'YlGn_r', 
                    vmax = 0.1,
                    vmin = 0,
                    ax = ax_p, 
                    annot = True, 
                    xticklabels = False,
                    yticklabels = False,
                    cbar_kws = {'aspect':round(len(pearson_r)*(3/4)),'pad': 0.01}
                    )
        plt.title('p-values', fontsize = 11)
        
        fig.suptitle(fig_title, y = 0.94) # Superior title
        
        plt.subplots_adjust(wspace=0.4)
        plt.show()
    
    if ret:
        return x,y,pearson_r

