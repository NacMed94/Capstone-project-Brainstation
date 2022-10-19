import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        if titles == None or i >= len(titles):
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
        
  