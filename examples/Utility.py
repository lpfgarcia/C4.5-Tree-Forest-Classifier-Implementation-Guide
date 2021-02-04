import matplotlib.pyplot as plt
def print_tree(tree,level=0,x=(0,1),y=(0,1),genfigures=False):
    lx=gx=x
    ly=gy=y
    if not tree[2]:
        print('\t'*level+f"It's {tree[0]}")
        if genfigures:
            plt.text(((x[0]+x[1])/2)*20-10, ((y[0]+y[1])/2)*20-10, tree[0], bbox=dict(facecolor=tree[0], alpha=0.5),verticalalignment='center',horizontalalignment='center')
    else:
        print('\t'*level+f"Split by \"{tree[0]}\""+(f" at {round(tree[1],4)}" if tree[1] else ''))
        if tree[1]:
            if genfigures:
                if tree[0]=='x':
                    gx = ((tree[1]+10)/20,x[1])
                    lx = (x[0],(tree[1]+10)/20)
                    plt.axvline(tree[1],y[0],y[1])
                elif tree[0]=='y':
                    gy = ((tree[1]+10)/20,y[1])
                    ly = (y[0],(tree[1]+10)/20)
                    plt.axhline(tree[1],x[0],x[1])
            print('\t'*level+f"If lesser than or equal to {round(tree[1],4)}:")
            print_tree(tree[2]['lessereq'],level+1,lx,ly,genfigures=genfigures)
            print('\t'*level+f"Otherwise ({tree[0]}>{round(tree[1],4)}):")
            print_tree(tree[2]['greater'],level+1,gx,gy,genfigures=genfigures)
        else:
            for key in tree[2]:
                print('\t'*level+f"If equal to {key}:")
                print_tree(tree[2][key],level+1,genfigures=genfigures)