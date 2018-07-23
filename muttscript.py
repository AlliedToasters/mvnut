import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis, KernelPCA
from sklearn.cluster import MeanShift
import urlmarker
import matplotlib.pyplot as plt
import random
from matplotlib import colors as mcolors




def bps_encode(bps_value):
    """Returns an encoding for an input bps value."""
    encoder = {
        'Content': 0,
        '+Optimize': 1,
        '+Photo': 2,
        'BlogMutt Complete': 3
    }
    return encoder[bps_value]

def info_count(other_info):
    """Takes other info field and returns number characters."""
    if isinstance(other_info, type(float())):
        return 0
    return len(other_info)

def url_finder(other_info):
    """Takes other info field and looks for url. Returns 0 or 1."""
    try:
        result = bool(re.findall(urlmarker.WEB_URL_REGEX, other_info))
    except:
        return 0
    if result:
        return 1
    else:
        return 0
    
def is_annual(plan_name):
    """Parses plan name to see if annual or not"""
    if 'Annual' in plan_name:
        return 1
    else:
        return 0
    
def is_content(plan_name):
    """Parses plan name to see if basic or content"""
    if 'Basic' in plan_name:
        return 0
    elif 'Content' in plan_name:
        return 1
    else:
        print('Unable to parse plan name: ', plan_name)
        raise Exception
        
def encode_state(state):
    """Encodes state with a value"""
    encoder = {
        'active':0,
        'pending':1,
        'expired':2,
        'canceled':3
    }
    return encoder[state]

def to_numeric(df):
    """Takes the subs dataframe and returns the numeric version."""
    cols = [
        'id', 
        'bps_plan', 
        'submitted_posts', 
        'purchased_posts', 
        'declined_posts', 
        'info_chars', 
        'has_url', 
        'plan_words', 
        'annual', 
        'content',
        'preferred_writers', 
        'state', 
        'lt_days'
    ]
    fts = pd.DataFrame(columns=cols)
    fts['id'] = df.id
    fts['bps_plan'] = df['bps_plan'].apply(bps_encode)
    fts['submitted_posts'] = df.submitted_posts
    fts['purchased_posts'] = df.purchased_posts
    fts['declined_posts'] = df.declined_posts
    fts['info_chars'] = df.other_info.apply(info_count)
    fts['has_url'] = df.other_info.apply(url_finder)
    fts['plan_words'] = df.plan_words
    fts['annual'] = df.plan_name.apply(is_annual)
    fts['content'] = df.plan_name.apply(is_content)
    fts['preferred_writers'] = df.preferred_writers
    fts['state'] = df.state.apply(encode_state)
    fts['lt_days'] = df.current_lifetime_days
    return fts

def plot_industries(dfs):
    """Bar plot of industry representation"""
    industries = pd.DataFrame(index=dfs[0].primary_industry.value_counts().index)
    industries['count0'] = dfs[0].primary_industry.value_counts().values
    for i in range(len(dfs)):
        if i==0:
            pass
        else:
            idcs = []
            vals = []
            for ind in list(industries.index):
                idcs.append(ind)
                vals.append(len(dfs[i][dfs[i].primary_industry==ind]))
            for idc, val in zip(idcs, vals):
                industries.at[idc, 'count{}'.format(i)] = val
    lbls = list(industries.index)
    countss = []
    for i in range(len(dfs)):
        counts = industries['count{}'.format(i)]
        countss.append(counts)
    ind = np.arange(len(industries))
    width = .35
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle('Cluster Representation by Industry')
    rects1 = ax.bar(ind, industries['count0'], width, color='green', label='all subscriptions')
    rects2 = ax.bar(ind+width, industries['count1'], width, color='orange', label='cluster')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(list(industries.index), rotation=90)
    return

def plot_word_counts(df):
    """Plots histograms of the number of words in subscription"""
    plan = pd.DataFrame(index=df.plan_words.astype(int).value_counts().index)
    plan['count'] = df.plan_words.astype(int).value_counts().values
    plan = plan.sort_index()
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.suptitle('Words in Plan')
    lbls = list(plan.index)
    bars = list(plan['count'])
    if len(bars) < 2:
        lbls.append(900)
        bars.append(0)
    xmin, xmax = plt.xlim()
    width = (xmin - xmax)/10
    ax.bar(lbls, bars, width=100, color='orange')
    
def describe_cluster(df, cluster):
    """Performs some statistics on a cluster"""
    clust = df[df.cluster==cluster]
    pop = df[~df.index.isin(clust.index)]
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Cluster {} Statistics'.format(cluster+1))
    ax[0, 0].set_title('Cluster Size: {}'.format(len(clust)))
    x = [len(clust), len(pop)]
    colors = ['orange', 'green']
    lbls = ['{}% in cluster'.format(round(100*(len(clust)/len(df)), 2)), '']
    ax[0, 0].pie(x, labels=lbls, startangle=180, colors=colors)
    ax[0, 1].set_title('Mean Accept Rate')
    lbls = []
    vals = []
    lbls.append('cluster')
    vals.append(clust.accept_rate.mean())
    lbls.append('overall')
    vals.append(df.accept_rate.mean())
    ax[0, 1].bar(lbls, vals, color=colors)
    ax[0, 1].set_ylabel('percent purchased')
    ax[0, 1].set_ylim(.6, 1)
    ax[0, 2].set_title('Average Lifetime Days')
    lbls = []
    vals = []
    lbls.append('cluster')
    vals.append(clust.current_lifetime_days.mean())
    lbls.append('overall')
    vals.append(df.current_lifetime_days.mean())
    ax[0, 2].bar(lbls, vals, color=colors)
    ax[1, 0].set_title('Mean PWF (days)')
    vals=[]
    vals.append(clust[clust.days_writer>0]['days_writer'].mean())
    vals.append(df[df.days_writer>0]['days_writer'].mean())
    ax[1,0].bar(lbls, vals, color=colors)
    ax[1, 0].set_ylabel('days/preferred writer')
    vals = []
    vals.append(clust[clust.posts_writer>0].posts_writer.mean())
    vals.append(df[df.posts_writer>0].posts_writer.mean())
    ax[1, 2].bar(lbls, vals, color=colors)
    ax[1,2].set_title('Mean PWF (posts)')
    ax[1,2].set_ylabel('posts/preferred_writer')
    ax[1, 1].axis('off')
    plt.show();
    
    plot_industries([df, clust])
    plt.show();
    plot_word_counts(clust)
    plt.show();
    


def revert_to_features(numeric):
    """Reverses numerization process."""
    cols = [
        'bps_plan', 
        'submitted_posts', 
        'purchased_posts', 
        'declined_posts', 
        'info_chars', 
        'has_url', 
        'plan_words', 
        'annual', 
        'content',
        'preferred_writers', 
        'state', 
        'lt_days'
    ]
    fts = pd.DataFrame(columns=cols)
    #bps_plan = numeric[]
    fts['bps_plan'] = df['bps_plan'].apply(bps_encode)
    fts['submitted_posts'] = df.submitted_posts
    fts['purchased_posts'] = df.purchased_posts
    fts['declined_posts'] = df.declined_posts
    fts['info_chars'] = df.other_info.apply(info_count)
    fts['has_url'] = df.other_info.apply(url_finder)
    fts['plan_words'] = df.plan_words
    fts['annual'] = df.plan_name.apply(is_annual)
    fts['content'] = df.plan_name.apply(is_content)
    fts['preferred_writers'] = df.preferred_writers
    fts['state'] = df.state.apply(encode_state)
    fts['lt_days'] = df.current_lifetime_days
    return numeric

def plot_days_industry(df):
    """Breaks down the average number of days to acquire 
    a new preferred writer by industry
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    df = df[df.days_writer>0].copy()
    mn = df.days_writer.mean()
    ax.axhline(y=mn, color='black', linestyle='dotted', label='average ({})'.format(round(mn,2)))
    industries = []
    means = []
    for ind in df.primary_industry.unique():
        industries.append(ind)
        if pd.isnull(ind):
            industries[-1] = 'n/a'
        means.append(df[df.primary_industry==ind].days_writer.mean())
    org_df = pd.DataFrame()
    org_df['mean_'] = means
    org_df['industry'] = industries
    org_df = org_df.sort_values('mean_', ascending=True)
    idx = list(org_df.index)[-1]
    org_df.drop(idx, inplace=True)
    ax.bar(org_df.industry, org_df.mean_)
    ax.set_xticklabels(org_df.industry, rotation=90)
    plt.title('Writer Preference Frequency per Days Active, by Industry')
    plt.legend();
    plt.show();
    return

def plot_posts_industry(df):
    """Breaks down the average number of days to acquire 
    a new preferred writer by industry
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    df = df[df.posts_writer>0].copy()
    mn = df.posts_writer.mean()
    ax.axhline(y=mn, color='black', linestyle='dotted', label='average ({})'.format(round(mn,2)))
    industries = []
    means = []
    for ind in df.primary_industry.unique():
        industries.append(ind)
        if pd.isnull(ind):
            industries[-1] = 'n/a'
        means.append(df[df.primary_industry==ind].posts_writer.mean())
    org_df = pd.DataFrame()
    org_df['mean_'] = means
    org_df['industry'] = industries
    org_df = org_df.sort_values('mean_', ascending=True)
    idx = list(org_df.index)[-1]
    org_df.drop(idx, inplace=True)
    ax.bar(org_df.industry, org_df.mean_)
    ax.set_xticklabels(org_df.industry, rotation=90)
    plt.title('Writer Preference Frequency per Post Purchased, by Industry')
    plt.legend();
    plt.show();
    return

def prepare_and_split(datapath):
    """Loads and splits the data. datapath is path to 
    csv file.
    """
    subs = pd.read_csv(datapath)
    subs['days_writer'] = np.where(subs.preferred_writers>0, subs.current_lifetime_days/subs.preferred_writers, 0)
    subs['posts_writer']=np.where(subs.preferred_writers>0, subs.purchased_posts/subs.preferred_writers, 0)
    train = subs.sample(frac=.5, random_state=1111)
    test = subs[~subs.index.isin(train.index)]
    
    plot_posts_industry(subs)
    plot_days_industry(subs)
    return train, test

def do_clustering(train, test, bandwidth=.7):
    """Does clustering given train and test dataframes."""

    
    #Convert features to numeric values with encodings
    X_train = to_numeric(train)
    X_test = to_numeric(test)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    
    kpca = KernelPCA(kernel='rbf', n_components=25, fit_inverse_transform=True)
    X = kpca.fit_transform(X_tr)
    X_ = kpca.transform(X_te)
    
    plt.scatter(X[:, 0], X[:, 1], label='train')
    plt.scatter(X_[:, 0], X_[:, 1], label='test')
    plt.legend();
    plt.show();
    
    mns = MeanShift(bandwidth=bandwidth)
    clusts = mns.fit_predict(X)
    clusts_ = mns.predict(X_)
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    for i, clust in enumerate(range(len(np.unique(clusts)))):
        random.seed(i)
        colors_ = list(colors.keys())
        clr = random.choice(colors_)
        name = 'cluster {}'.format(clust)
        members = np.where(clusts==clust, True, False)
        print('cluster {} (train) size:'.format(name), np.where(members, 1, 0).sum())
        plt.scatter(X[members, 0], X[members, 1], label=name, color=clr)
        members = np.where(clusts_==clust, True, False)
        print('cluster {} (test) size:'.format(name), np.where(members, 1, 0).sum())
        plt.scatter(X_[members, 0], X_[members, 1], color=clr)

    plt.xlabel('principal component 1')
    plt.ylabel('principal component 2')
    
    plt.legend();
    plt.show();
    
    train['cluster'] = clusts
    test['cluster'] = clusts_
    
    df = pd.concat([train, test], axis=0)
    df = df.sort_index().copy()
    df['posts/day'] = np.where(df.current_lifetime_days > 0, df.submitted_posts/df.current_lifetime_days, 0)
    df['has_url'] = X_train.has_url
    df['has_url'] = X_test.has_url
    df['info_chars'] = X_train.info_chars
    df['info_chars'] = X_test.info_chars
    df['accept_rate'] = df.decline_rate * (-1) + 1
    
    return df