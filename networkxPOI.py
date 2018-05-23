# libraries
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# Build a dataframe with 4 connections
df = pd.DataFrame({ 'from':['Query3','James Steffes','Query8','Davis Pete','Query8','Chris G','Query8','Mark Taylor','Query8','Sara Shackleton','Query8','Tana Jones','Query8','Kay Mann','Query3','Jeff Dasovich','Query3','Steven Kean','Query2','Davis Pete','Query2','Chris G','Query2','Mark Taylor','Query2','Sara Shackleton','Query2','Tana Jones','Query2','Kay Mann','Query5','Davis Pete','Query5','Chris G','Query5','Mark Taylor','Query5','Sara Shackleton','Query5','Tana Jones','Query5','Kay Mann','Query4','Steven Kean','Query4','Jeff Dasovich','Query7','Jeff Skillings','Query3','Kenneth Lay','Query1','Sara Shackleton','Query1','Kenneth Lay','Query1','Jeff Skillings','Query1','James Steffes', 'Query2','Jeff Skillings','Query2','Sara Shackleton', 'Query4','Richard Shapiro', 'Query3','Kate Syemmes', 'Query5','Sara Shackleton', 'Query6','Sara Shackleton', 'Query6','Mark Taylor','Query6','Jeff Dasovich','Query6','Steven Kean','Query6','Sara Shackleton','Query7','Sara Shackleton', 'Query7','Davis Pete', 'Query7','Chris G','Query7','Mark Taylor','Query7','Kenneth Lay','Query7','Kay Mann','Query7','Tana Jones', 'Query8','Kenneth Lay','Query8', 'Jeff Dasovich', 'Query8', 'Mark Taylor','Query8', 'Kay Mann', 'Query8', 'Sara Shackleton', 'Query8', 'James Steffes','Query9', 'Jeff Skillings','Query9', 'Kay Mann', 'Query9', 'Tana Jones','Query9', 'Mark Taylor','Query9', 'Sara Shackleton', 'Query9', 'Davis Pete', 'Query9', 'Chris G', 'Query10', 'Chris G','Query10', 'Mark Taylor','Query10', 'Sara Shackleton','Query10', 'Tana Jones','Query10', 'Kay Mann'],
                    'to':['James Steffes', 'Query3','Davis Pete','Query8','Chris G','Query8','Mark Taylor','Query8','Sara Shackleton','Query8','Tana Jones','Query8','Kay Mann','Query8','Query3','Jeff Dasovich','Query3','Steven Kean','Davis Pete','Query2','Chris G','Query2','Mark Taylor','Query2','Sara Shackleton','Query2','Tana Jones','Query2','Kay Mann','Query2','Davis Pete','Query5','Chris G','Query5','Mark Taylor','Query5','Sara Shackleton','Query5','Tana Jones','Query5','Kay Mann','Query5','Steven Kean','Query4','Jeff Dasovich','Query4','Jeff Skillings','Query7','Kenneth Lay','Query3','Sara Shackleton','Query1','Kenneth Lay','Query1','Jeff Skillings','Query1','James Steffes','Query1', 'Jeff Skillings','Query2','Sara Shackleton','Query2', 'Richard Shapiro','Query4', 'Kate Syemmes', 'Query3','Sara Shackleton', 'Query5', 'Sara Shackleton','Query6', 'Mark Taylor','Query6','Jeff Dasovich','Query6','Steven Kean','Query6','Sara Shackleton','Query6', 'Sara Shackleton','Query7', 'Davis Pete','Query7', 'Chris G','Query7', 'Mark Taylor','Query7','Kenneth Lay','Query7','Kay Mann','Query7','Tana Jones','Query7','Kenneth Lay','Query8', 'Query8','Jeff Dasovich',  'Query8', 'Mark Taylor','Query8','Kay Mann',  'Query8','Sara Shackleton', 'Query8', 'James Steffes',  'Jeff Skillings','Query9', 'Kay Mann', 'Query9', 'Tana Jones','Query9', 'Mark Taylor','Query9', 'Sara Shackleton', 'Query9', 'Davis Pete','Query9',  'Chris G', 'Query9',  'Chris G','Query10', 'Mark Taylor','Query10','Sara Shackleton','Query10',  'Tana Jones', 'Query10', 'Kay Mann','Query10']})

# And a data frame with characteristics for your nodes
carac = pd.DataFrame ({ 'ID':['Query1', 'Query2','Query3','Query4','Query5', 'Query6', 'Query7', 'Query8','Query9', 'Query10'], 'myvalue':['group2','group2','group3','group4','group5', 'group6', 'group7', 'group8', 'group9',  'group10'] })

# Build your graph
G=nx.from_pandas_dataframe(df, 'from', 'to',create_using=nx.Graph())
# The order of the node for networkX is the following order:
G.nodes()
# Thus, we cannot give directly the 'myvalue' column to netowrkX, we need to arrange the order!

# Here is the tricky part: I need to reorder carac to assign the good color to each node
carac= carac.set_index('ID')
carac=carac.reindex(G.nodes())

# And I need to transform my categorical column in a numerical value: group1->1, group2->2...
carac['myvalue']=pd.Categorical(carac['myvalue'])
carac['myvalue'].cat.codes

# Custom the nodes:
nx.draw(G, with_labels=True, node_color=carac['myvalue'].cat.codes, cmap=plt.cm.Set1, node_size=1500)


plt.show()
































# Build a dataframe with 4 connections
df = pd.DataFrame({ 'from':['Jeff Dasovich','Director','Jeff Dasovich','Not Active','Tana Jones', 'Employee', 'Tana Jones','Active','Kate Syemes','Employee','Kate Syemes','Averagely Active','Chris G','Employee','Chris G','Active', 'Davis Pete','Employee','Davis Pete','Active','Mark Taylor','Employee','Mark Taylor','Active','James Steffes','Active','James Steffes','VP','Tana Jones','Employee','Active','Tana Jones','Active','Kay Mann','Kay Mann','Employee','Sara Shackleton','Active','VP', 'Sara Shackleton','VP','Steven Kean','Not Active','Steven Kean','Not Active','Richard Shapiro','Jeff Skillings','Active','VP', 'Steven Kean','VP','Richard Shapiro','Active','Jeff Skillings','Kenneth Lay', 'CEO','Very Active', 'Kenneth Lay', 'Jeff Skillings','CEO'],
                    'to':['Director','Jeff Dasovich','Not Active','Jeff Dasovich','Employee','Tana Jones',  'Active','Tana Jones','Employee','Kate Syemes','Averagely Active','Kate Syemes','Employee','Chris G','Active','Chris G', 'Employee','Davis Pete','Active','Davis Pete','Employee','Mark Taylor','Active','Mark Taylor','Active','James Steffes','VP','James Steffes','Employee','Tana Jones','Tana Jones','Active','Kay Mann','Active','Employee','Kay Mann','Active','Sara Shackleton','Sara Shackleton','VP','Steven Kean','Not Active','Steven Kean','VP','Richard Shapiro','Not Active','Active','Jeff Skillings','Steven Kean','VP','Richard Shapiro','VP','Jeff Skillings','Active','Very Active','Kenneth Lay','CEO', 'Kenneth Lay', 'CEO','Jeff Skillings']})

# And a data frame with characteristics for your nodes
carac = pd.DataFrame ({ 'ID':['Active', 'Director','Averagely Active','Not Active', 'CEO','VP', 'Very Active','Employee'], 'myvalue':['group1','group2','group3','group4','group5','group6','group7', 'group8'] })

# Build your graph
G=nx.from_pandas_dataframe(df, 'from', 'to',create_using=nx.Graph())
# The order of the node for networkX is the following order:
G.nodes()
# Thus, we cannot give directly the 'myvalue' column to netowrkX, we need to arrange the order!

# Here is the tricky part: I need to reorder carac to assign the good color to each node
carac= carac.set_index('ID')
carac=carac.reindex(G.nodes())

# And I need to transform my categorical column in a numerical value: group1->1, group2->2...
carac['myvalue']=pd.Categorical(carac['myvalue'])
carac['myvalue'].cat.codes

# Custom the nodes:
nx.draw(G, with_labels=True, node_color=carac['myvalue'].cat.codes, cmap=plt.cm.Set1, node_size=1500)


plt.show()

