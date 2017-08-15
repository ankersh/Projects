from graph_tool.all import *
import graph_tool.all as gt
import sys
from gi.repository import Gtk, Gdk, GdkPixbuf, GObject
import numpy as np
import random
import math



def newBiRegGraph(n):       #Creates a 3-regular bipartite graph with no double edges.
    edgeE = np.zeros((n,n))     #This is an incident matrix, edgeE[x,y] = k where k is the number of edges between x and y. Obviously we need k = 1 for this graph.
    g = Graph(directed=False)
    g.add_vertex(n)
    edgelist = []           #Will contains edge of the graph in the form [x1,y1,x2,y2,x3,y3...]
    needEdges = []          #Set of vertices from (0,49) which don't yet have a degree of 3 (we will refer to these as vertices on the left hand side)
    deg0 = []
    deg1 = []               #These store all the vertices of degree 0, degree 1.. etc on vertices (50,99) (we will refer to these as the vertices on the right hand side)
    deg2 = []
    for i in range(n/2,n):
        deg0.append(i)
    for i in range(0,n/2):
        needEdges.append(i)

    while deg0:
        x = random.choice(needEdges)                    #We pick in total 50 edges, one for every vertex on the right hand side, giving them all a degree of 1 at the end.
        y = random.choice(deg0)
        if g.vertex(x).out_degree() == 3:               #If a vertex on the left has a degree of 3 it won't need any more edges so we remove it.
            needEdges.remove(x)
        else:
            edgeE[x,y] += 1
            edgelist.append(x)
            edgelist.append(y)
            g.add_edge(x,y)
            deg0.remove(y)
            deg1.append(y)

    while deg1:
        x = random.choice(needEdges)                #We pick 50 more edges, one for each vertex on the right hand side, giving them all a degree of 2 at the end.
        y = random.choice(deg1)
        if g.vertex(x).out_degree() == 3:
            needEdges.remove(x)
        else:
            edgeE[x,y] += 1
            edgelist.append(x)
            edgelist.append(y)
            g.add_edge(x,y)
            deg1.remove(y)
            deg2.append(y)

    while deg2:
        x = random.choice(needEdges)                 #We pick 50 more edges, one for each vertex on the right hand side, giving them all a degree of 3 at the end.
        y = random.choice(deg2)
        if g.vertex(x).out_degree() == 3:
            needEdges.remove(x)
        else:
            edgeE[x,y] += 1
            edgelist.append(x)
            edgelist.append(y)
            g.add_edge(x,y)
            deg2.remove(y)

    k = 0
    while (k < len(edgelist)):
        x = edgelist[k]                         #Unfortunately there is no way to create an edge in the graph only if it doesn't exist, so we might have double edges which will break our algorithm.
        y = edgelist[k+1]                       #We don't need to worry about triple edges as the algorithm will still work as the two vertices will become their own connected component.
        T = -1
        if edgeE[x,y] == 2:                     #If there are two edges between x and y, we will replace the edge.
            g.remove_edge(g.edge(x,y))          #Removes the edge (x,y) and sets edgeE to 1
            edgeE[x,y] += -1
            for i in range(0,50):
                if (edgeE[i,y] == 0):           #Finds a vertex on the left hand side that y is not connected to
                    T = g.vertex(i)             #Sets T to be this vertex
                    break
            V = -1
            for i in T.out_neighbours():        #Finds a vertex on the right hand side which T is connected to but X isn't. We know there must be at least 1 as both T and X have degree 3 but X isn't connected to T
                if edgeE[x,int(i)] == 0:
                    V = i                       #Sets V to be this vertex
                    break
            g.remove_edge(g.edge(T,V))         #Removes the edge (T,V)
            edgeE[int(T),int(V)] += -1
            g.add_edge(T,y)                    #Adds the edge (T,y)
            edgeE[(int(T),y)] += 1
            g.add_edge(x,V)                    #Adds the edge (x,V)
            edgeE[x,int(V)] += 1
        k += 2

    return g

def findSpanningForest(g):  #Finds a spanning forest of the given graph g

    t = Graph(directed=True)
    t.add_vertex(g.num_vertices())
    treeV = []                          #These are the vertices stored in the spanning forest
    Queue = []                          #This is the queue of vertices which we have found but not searched
    notfound = []                          #These are the vertices which have not been found. Once this is empty, all the vertices are in the spanning forest.
    for i in range(0,g.num_vertices()):
        notfound.append(i)
    startVertex = random.randint(0,(g.num_vertices()-1))        #Randomly picks a vertex to start the search from
    Queue.append(startVertex)
    treeV.append(startVertex)
    notfound.remove(startVertex)
    while notfound:
        while Queue:
            for v in Queue:                     #Picks the first vertex in the Queue
                neighbours = g.get_out_neighbours(v)
                for y in neighbours:
                    if not (y in treeV):                #If y isn't in the forest already, add it to the forest and add it to the queue of vertices to be explored.
                        treeV.append(y)
                        notfound.remove(y)
                        Queue.append(y)
                        t.add_edge(v,y)
                Queue.remove(v)

        if notfound:            #If there are still unfound vertices then they lie in a different connected component. We start the search again from one of these.
            i = notfound[0]
            notfound.remove(i)
            treeV.append(i)
            Queue.append(i)

    return t

def findFCycle(T,u,v):          #Finds a fundamental cycle between nodes U and V in T
    F = Graph(directed=False)           #This graph will store all the edges of the fundamental cycle.
    F.add_vertex(T.num_vertices())
    x = T.vertex(u)
    y = T.vertex(v)
    root = x                #The 'root' variable will be used to store the lowest common ancestor of u and v, the root of the smallest subtree with them both contained.
    F.add_edge(x,y)
    xVertex = [x]           #We will store all the ancestors of X here.
    yVertex = [y]           #We will store all the ancestors of Y here. The first time we find an ancestor in both, we will have the lowest common ancestor.

    while (int(x) != int(y)):               #When int(x) = int(y) they will equal the lowest common ancestor
        for px in x.in_neighbours():        #If empty will do nothing
            ParentX = px                    #As we have a directed spanning forest, the only in vertex will be the parent of X. Clearly there is only one, but it needs to be called as a list.
        for py in y.in_neighbours():
            ParentY = py

        if ParentX in yVertex:            #If the parent of X is in the list of ancestors of Y, it is a common ancestor. As we break the loop the first time it happens, it's the lowest common ancestor.
            root = ParentX
            xVertex.append(ParentX)
            break
        if ParentY in xVertex:          #If the parent of Y is in the list of ancestors of Y, it is a common ancestor. As we break the loop the first time it happens, it's the lowest common ancestor.
            root = ParentY
            yVertex.append(ParentY)
            break
        xVertex.append(ParentX)         #If we havent found a common ancestor, add these both to the list and then continue looking at the parent vertices.
        yVertex.append(ParentY)
        x = ParentX
        y = ParentY
        if x == y:                          #If x and y start on exactly the same level with respects to the parent vertex, we will find the lowest common ancestor for the first time here.
            root = x

    index = xVertex.index(root)             #We want to know where abouts the root vertex occurs. It might be the first ancestor of x for example, but the 10th of y, in which case we'd have 9 vertices in xVertex
    while len(xVertex) > index+1:                   #which aren't in the fundamental cycle.
        del xVertex[index+1]
    index = yVertex.index(root)
    while len(yVertex) > index+1:
        del yVertex[index+1]

    for i in range (0,(len(yVertex)-1)):            #Now yVertex and xVertex contain all the vertices in the path to the lowest common ancestor, so we create edges between each of these vertices, giving us the fundamental cycle.
        F.add_edge(yVertex[i],yVertex[i+1])
    for i in range (0,(len(xVertex)-1)):
        F.add_edge(xVertex[i],xVertex[i+1])



    return F

def edgesInTree(Gr,Tr):         #Returns a set of edges that are in G but not in its spanning tree
    TEdges = []             #Stores the edges in Tr
    GEdges = []             #Stores the edges in Gr
    IndEdges = []                   #This will store a list of fundamental edges.
    i = 0
    for edge in Tr.edges():
        x = edge.source()
        y = edge.target()
        TEdges.append(int(x))
        TEdges.append(int(y))           #The graph library won't check edges are in a graph correctly, so we instead save them to two lists and iterate over them.
    for edge in Gr.edges():             #Slightly time consuming but the only way to guarentee it works.
        x = edge.source()
        y = edge.target()
        GEdges.append(int(x))
        GEdges.append(int(y))

    while i < len(GEdges):
        x = GEdges[i]
        y = GEdges[i+1]
        j = 0
        flag = 0
        while j < len(TEdges):
            if((x == TEdges[j]) & (y == TEdges[j+1])):          #Given (x,y) in Gr, if (x,y) or (y,x) is in Tr then set the flag to 1 so we know not to include it as a fundamental edge.
                flag = 1
                break
            elif((y == TEdges[j]) & (x == TEdges[j+1])):
                flag = 1
                break
            j += 2

        if flag == 0:               #If flag == 0 then the edge (x,y) isn't in Tr so it's a fundamental edge.
            IndEdges.append(int(x))
            IndEdges.append(int(y))
        i += 2

    return IndEdges

def FindCycle(G,v):             #Searches for a cycle in G containing V using a depth first search.
    n = G.num_vertices()
    v = G.vertex(v)
    edgecount = np.zeros((n,n))
    c = []
    cycle = 0                   #Flag for telling us if we found a cycle or not
    found = []                  #found[x] = -1 if x hasn't yet been found, found[x] = y if x was found by y.
    for i in range(0,n):
        found.append(-1)
    queue = [v]
    while queue:
        x = queue[0]
        queue.remove(x)
        for y in x.all_neighbours():
            if edgecount[int(x),int(y)] == 0:               #If the edge (x,y) has not been visited before
                if found[int(y)] == int(v):                 #We are checking to see if a neighbouring vertex of x, y, has already been found by V. If so clearly this gives us a cycle.
                    v1 = x                                  #We will construct the cycle from these two vertices.
                    v2 = y
                    cycle = 1
                    del queue[:]
                else:                                       #If not we continue searching
                    edgecount[int(x),int(y)] += 1           #Mark an edge between x and y so we know the edge (x,y) has been visited.
                    edgecount[int(y),int(x)] += 1
                    found[int(y)] = int(x)
                    queue.insert(0,y)                   #Insert y to the top of the queue and search from there.
    if cycle == 1:                          #If we have found a cycle, we will construct it in a list in the form [v1,y1,y2...v,v2,v1] where yi is a vertex in the cycle.
        x = int(v1)
        y = found[x]
        c.append(x)
        while not x == int(v):
            c.append(y)
            x = y                       #To construct this list we will use the Found list until we go back to the original vertex v
            y = found[x]
        c.append(int(v2))               #Finally we add v2 (a neighbouring vertex of v which we used to determine a cycle existed) and v1. This gives us the consecutive vertices in the cycle.
        c.append(int(v1))




    return c

def AltFindCycle(G,v):             #Searches for a cycle in G containing V. This puts the edges in such a format that finding a perfect matching in the complement is simple.
    n = G.num_vertices()
    v = G.vertex(v)
    found = []                  #Found[v] returns the node that found v
    visited = []                #Visited[v] tells us if v has been visited yet or not
    c1 = []                      #Stores the two halves of the cycle if a cycle exists
    c2 = []
    c = []
    cycle = 0                   #Flag for telling us if we found a cycle or not


    for i in range(0,n):
        found.append(-1)         #Sets found[i] = 0 for all vertices i in G
        visited.append(0)
    Queue = [v]
    visited[int(v)] = 1
    while Queue:
        x = Queue[0]
        Queue.remove(x)
        for y in x.all_neighbours():
            if visited[int(y)] == 0:
                Queue.append(y)
                found[int(y)] = int(x)
                visited[int(y)] = 1
            else:
                if not (found[int(x)] == int(y)):         #If x was not originally found by y but y has already been visited, we have a cycle.
                    cycle = 1
                    del Queue[:]
                    v1 = int(x)
                    v2 = int(y)

    if cycle == 1:
        x = v1                  #Our cycle was found at the edge (v1,v2). We backtrack from v1 to V and store this half of the cycle in the list c1.
        y = found[x]
        c1.append(x)
        c1.append(y)
        while not(y == int(v)):
            x = found[x]
            y = found[x]
            c1.append(y)
        x = v2
        y = found[x]        #We then start at v2 and backtrack from v2 to V and store this half of the cycle in the list c2.
        c2.append(x)
        c2.append(y)
        while not(y == int(v)):
            x = found[x]
            y = found[x]
            c2.append(y)

        k = len(c1)-1
        while k >= 0:           #We merge the two halves of the cycle into one list, c.
            c.append(c1[k])
            k += -1

        k = 0
        while k < len(c2):
            c.append(c2[k])
            k += 1

    return c

def VInduce(G,T):               #Returns the graph induced by G over vertices  T
    TList = []              #Stores a list of vertices that aren't isolated in T.
    for edge in T.edges():
        x = int(edge.source())
        y = int(edge.target())
        if not x in TList:
            TList.append(x)
        if not y in TList:
            TList.append(y)

    v = Graph(directed=False)           #Stores the induced graph.
    v.add_vertex(G.num_vertices())

    for edge in G.edges():          #For every edge (u,v) in G, if both (u,v) are in T then add the edge to the induced graph.
        x = int(edge.source())
        y = int(edge.target())
        if (x in TList) & (y in TList):
            v.add_edge(x,y)

    return v

def FindCC(G):              #Finds and returns a list containing connected components
    CList = []          #List that components will be stored in
    GList = []          #List of vertices in G and havent been found in a connected component yet.
    visited = []        #Marks if vertex has been visited or not

    n = G.num_vertices()
    edgecountC = np.zeros((n,n))
    for i in range(0,n):
        GList.append(i)
        visited.append(0)               #visited[u] marks if the vertex u has been visited or not.

    Queue = [G.vertex(1)]       #Random vertex to start search from.
    visited[1] = 1
    GList.remove(1)
    while GList:
        f = Graph(directed=False)               #This graph will store a connected component.
        f.add_vertex(g.num_vertices())
        while Queue:
            x = Queue[0]                #Very similar to our spanning forest algorithm.
            Queue.remove(x)
            for y in x.all_neighbours():
                if visited[int(y)] == 0:            #If a neighbour of x, y, hasn't been visted yet, remove it from Glist and mark it as visited.
                    Queue.append(y)
                    visited[int(y)] = 1
                    GList.remove(int(y))
                if edgecountC[int(x),int(y)] == 0:              #If we currently don't have an edge existing between x and y then add one.
                    if edgecountC[int(y),int(x)] == 0:
                        f.add_edge(x,y)
                        edgecountC[int(x),int(y)] += 1


        if GList:
            Queue.append(G.vertex(GList[0]))            #If GList then there is still another connected component, we will start again at a random vertex and find this component.
            visited[GList[0]] = 1
            GList.remove(GList[0])
        CList.append(f)

    return CList

def isInduced(T,g):           #Returns True if T is induced over G, false otherwise
    n = g.num_vertices()
    TEdge = np.zeros((n,n))
    TVertices = []              #Stores the vertices in T
    for edge in T.edges():
        x = int(edge.source())
        y = int(edge.target())
        TEdge[x,y] += 1                 #A matrix which stores all the edges in T.
        TEdge[y,x] += 1
        if not x in TVertices:
            TVertices.append(x)
        if not y in TVertices:
            TVertices.append(y)
    for edge in g.edges():
        x = int(edge.source())
        y = int(edge.target())
        if (x in TVertices) & (y in TVertices):             #Checks every edge in G to see if both vertices are in T but the edge isn't, which would mean T isn't induced.
            if TEdge[x,y] == 0:
                return False

    return True

def numDescendants(g,v,visited,descendants):          #Returns an array with the number of descendants each vertex in g has.
    for y in v.out_neighbours():
        if visited[int(y)] == 0:
            visited[int(v)] = 1
            descendants = numDescendants(g,y,visited,descendants)          #Recursively calculates the number of descendants for every vertex in g.
            descendants[int(v)] += descendants[int(y)]

    return descendants

def FindCutVertex(T,root):          #Returns a 1/3 - 2/3 cut vertex as defined in notes
    descendants = []
    visited = []                #Stores if a vertex has been visited or not.
    for i in range(0,T.num_vertices()):
        descendants.append(1)
        visited.append(0)
    descendants = numDescendants(T,root,visited,descendants)


    k = float(T.num_edges() + 1)
    lo = math.ceil((k-1)/3)                 #Calculates the maximum/minimum number of descendants required.
    hi = math.ceil(2*k/3)

    if k == 4:          #If we only have 4 vertices then we must pick the only 3-vertex as our cut vertex
        for v in T.vertices():
            if v.out_degree() == 3:
                return v

    else:
        for v in T.vertices():
            if (descendants[int(v)] >= lo)&(descendants[int(v)] <= hi):     #If this is true then V is clearly a cut vertex.
                return v

def Complement(G,M):            #Returns the complement of M with respect to edges in g
    N = Graph(directed=False)
    N.add_vertex(G.num_vertices())

    for edgeG in G.edges():
        flag = 0
        x = int(edgeG.source())
        y = int(edgeG.target())
        for edgeM in M.edges():                 #If an edge is in G and it's in M, flag = 1 and we know not to add the edge to M
            xm = int(edgeM.source())
            ym = int(edgeM.target())
            if (((xm == x)&(ym==y))|((xm==y)&(ym==x))):
                flag = 1
        if flag == 0:           #If flag = 0 we know the edge should be in the complement so we add it.
            N.add_edge(x,y)
    return N

def matchingInComplement(N):      #Computes a perfect matching in the complement N
    H = Graph(directed=False)
    H.add_vertex(N.num_vertices())

    for cycle in FindCC(N):     #Every connected component of N is a cycle.
        if cycle.num_edges() > 0:
            for edge in cycle.edges():
                x = edge.source()
                break                   #This just finds a random vertex in the cycle.
            c = AltFindCycle(cycle,x)      #This gives us a nice list of edges in the cycle we can use to take alternating edges.
            count = 0
            i = 0
            while i < (len(c)-1):               #We add every second edge to H, as we know the cycle is even this gives us a perfect matching of the cycle.
                if count%2 == 0:
                    H.add_edge(c[i],c[i+1])
                count += 1
                i += 1
    return H

def XOREdge(G,e):                   #Performs G = G XOR E
    x = int(e.source())                 #Unfortunately there is no way to check if an edge exists in a graph already
    y = int(e.target())                 #IF we try to remove a non-existant edge, we get an error, meaning this method is slightly more complicated than it needs to be.
    flag = 0
    for edge in G.edges():              #We look through the edges in G, if e is in G we flag it and remove it, if not we add it.
        xg = int(edge.source())
        yg = int(edge.target())
        if (((xg == x)&(yg==y))|((xg==y)&(yg==x))):
            flag = 1
            RemoveE = edge
            break
    if flag == 0:
        G.add_edge(x,y)
        e = G.edge(x,y)
    if flag == 1:
        G.remove_edge(RemoveE)
    return G

def XOREdge2(G,x,y):                   #Performs G = G XOR E
    flag = 0
    Gg = G.copy()
    for edge in G.edges():
        xg = int(edge.source())
        yg = int(edge.target())
        if (((xg == x)&(yg==y))|((xg==y)&(yg==x))):
            flag = 1
            RemoveE = edge
            break
    if flag == 0:
    #    print "The edge (" + str(x) + ", " + str(y) + ") is not in M"
        Gg.add_edge(x,y)
    if flag == 1:
    #    print "The edge (" + str(x) + ", " + str(y) + ") is in M"
        Gg.remove_edge(RemoveE)
    return Gg

def Union(G,H):                     #Returns the union of G and H

    C = G.copy()
    for edge in H.edges():
        x = int(edge.source())
        y = int(edge.target())
        flag = 0
        for edgeG in C.edges():
            xg = int(edgeG.source())
            yg = int(edgeG.target())
            if (((xg == x)&(yg==y))|((xg==y)&(yg==x))):
                flag = 1
        if flag == 0:
            C.add_edge(x,y)

    return C


#######################      Main Algorithm Starts Here       ############################



def ppmToForest(g):     #SECTION 5: CONVERTING PPM TO FOREST

    num = g.num_vertices()
    edgecount = np.zeros((num,num))         #This will count how many times an edge appears in the fundamental cycle.
    m = g.copy()                #Initially our PPM is a copy of g.
    t = findSpanningForest(g)


    IndEdges = edgesInTree(g,t)
    if IndEdges:                #If there are any edges in G and not our spanning forest, these are our fundamental edges.
        i = 0
        while i < len(IndEdges):
            f = findFCycle(t,IndEdges[i],IndEdges[i+1])         #Find a fundamental cycle for the first independent edge.
            FundamentalCycles1.append(f)
            for edge in f.edges():
                FCEdges.append(edge)
                x = int(edge.source())
                y = int(edge.target())                  #For every edge in the fundamental cycle, add 1 to edgecount[x,y]
                edgecount[x,y] += 1
                edgecount[y,x] += 1

            i += 2
    for edge in g.edges():
        x = int(edge.source())
        y = int(edge.target())
        if edgecount[x,y]%2 == 1:               #If edgecount[x,y] is odd then it is included in a odd number of fundamental cycles and therefore it would be removed after xoring a set of fundamental cycles.
            m.remove_edge(edge)

    return m

def Induce(G,m):          #SECTION 6: FOREST TO INDUCED FOREST
    PPM = m.copy()
    notinduced = []
    for tree in FindCC(PPM):
        if isInduced(tree,G) == False:
            notinduced.append(tree)     #We store uninduced trees in a list
            ITrees.append(tree)         #We will store the uninduced trees in a second list which we will use for the visualisation afterwards.


    while notinduced:
        tree = notinduced[0]                #We select a uninduced tree
        notinduced.remove(tree)
        for v in tree.vertices():
            if v.out_degree() == 3:         #We select a random 3-vertex to be the root vertex
                root = v
                break
        v = FindCutVertex(tree,root)        #We find a cut vertex given this root
        GT = VInduce(G,tree)                #GT = the graph tree induced over G
        cycle = FindCycle(GT,int(v))        #We see if the cut vertex lies on a cycle in GT, the graph of T induced over G.
        if cycle:
            i = 0
            while i < len(cycle)-1:         #If we find a cycle, M = M XOR cycle.
                x = cycle[i]
                y = cycle[i+1]
                PPM = XOREdge2(PPM,x,y)
                i += 1

        else:
            tree2 = Graph(tree)             #If we have not found a cycle we need to split the tree up still
            sv = Graph(directed=False)          #SV = Every edge in tree which contains v
            sv.add_vertex(G.num_vertices())

            for edgek in tree.edges():          #tree2 = tree - {v}
                x = edgek.source()
                y = edgek.target()
                if (x == v)|(y == v):
                    tree2.remove_edge(edgek)
                    sv.add_edge(x,y)


            I = Induce(G,tree2)                 #I = tree2 induced over G

            edgesToRemove = []
            for edge in tree.edges():       #PPM = PPM - tree
                x = edge.source()
                y = edge.target()
                for edgem in PPM.edges():
                    xm = edgem.source()
                    ym = edgem.target()
                    if (((xm==x)&(ym==y))|((xm==y)&(ym==x))):
                        edgesToRemove.append(edgem)

            for edge in edgesToRemove:
                PPM.remove_edge(edge)



            PPM = Union(PPM,I)      #PPM = PPM U I


            PPM = Union(PPM,sv)     #PPM = PPM U SV


    return PPM

def IForestToPM(G,M):   #SECTION 7: INDUCED FOREST TO PERFECT MATCHING
    global iteration
    global twoThreeGraph

    PM = M.copy()                   #We take a copy of M and we will conver
    num = G.num_vertices()
    edgecount = np.zeros((num,num))
    n = Complement(G,M)                 #We find a complement of M and we find a matching in this complement.
    match = matchingInComplement(n)           #Note: At this stage this function WILL fail if the complement is not a set of disjoint even cycles
    h = Union(M,match)                          #Meaning that we cannot use this unless G is 3-regular and Bipartite.
    if iteration == 0:
        twoThreeGraph = h
    th = findSpanningForest(h)          #Find a spanning forest of h

    IndEdges = edgesInTree(h,th)            #Find a set of independent edges of h with respect to the spanning tree th
    if IndEdges:
        i = 0
        while i < len(IndEdges):                #For every independent edge, find the fundamental cycle
            f = findFCycle(th,IndEdges[i],IndEdges[i+1])
            FundamentalCycles2.append(f)
            for edge in f.edges():
                FCEdges2.append(edge)
                x = int(edge.source())
                y = int(edge.target())
                edgecount[x,y] += 1             #For every edge in the fundamental cycle, add 1 to edgecount[x,y]
                edgecount[y,x] += 1
            i += 2

    EdgesToXOR = []
    for edges in h.edges():
        x = int(edges.source())
        y = int(edges.target())
        if edgecount[x,y]%2 == 1:               #If edgecount[x,y] is odd then it is included in a odd number of fundamental cycles and therefore it would be removed after xoring a set of fundamental cycles.
            EdgesToXOR.append(edges)

    for K in EdgesToXOR:
        x = int(K.source())
        y = int(K.target())
        PM = XOREdge2(PM,x,y)

    iteration += 1
    for v in PM.vertices():
        if v.out_degree() != 1:
            PM = IForestToPM(G,PM)              #We may have to run this more than once, so we check to see if PM is a perfect matching yet or not.
    return PM




ITrees = []                         #Stores any tree which is uninduced
RemovedEdges = []                   #This stores all the uninduced edges

FCEdges = []                        #These store the edges that appear in the set of fundamental cycles found in ppmToForest
FundamentalCycles1 = []             #This stores the set of fundamental cycles found in ppmToForest

FCEdges2 = []                       #These store the edges that appear in the set of fundamental cycles found in IForestToPM
FundamentalCycles2 = []             #This stores the set of fundamental cycles found in ppmToForest

def checkPM():
    for v in pm.vertices():
        if v.out_degree() != 1:
            print "Error, vertex with a degree not equal to 1"              #This should never actually happen because IForestToPM will only stop running when every vertex has a degree of 1
                                                                            #but it's still good to check.
    for edge in pm.edges():
        flag = 0
        x = int(edge.source())
        y = int(edge.target())
        x = g.vertex(x)
        for z in x.all_neighbours():                        #This checks to make sure every edge in the perfect matching is a edge in G.
            if int(z) == y:
                flag = 1
        if flag == 0:
            print "Error, edge not in G but in perfect matching!"




iteration = 0                       #We use this so we can take a copy of the twoThreeGraph the first time we run IForestToPM

Size = input("Please enter the number of vertices in the graph (Larger graphs may take more time to render): ")
print ""

g = newBiRegGraph(Size)              #g = 3-regular bipartite graph
twoThreeGraph = g                   #After we run IForestToPM this will be a copy of the twoThreeGraph H = (m U Comp)
m = ppmToForest(g)                  #m = odd forest of g
ifo = Induce(g,m)                   #ifo = induced odd forest of g
pm = IForestToPM(g,ifo)             #pm = perfect matching of g
Comp = Complement(g,ifo)            #The complement of the induced forest over g



checkPM()           #Runs checkPM to make sure pm is definitely a correct perfect matching of g.



############################## Visualisations ####################################



totalcounter = 0          #Counts the number of edges we have added/removed from the graph in total.
cyclecounter = 0          #Counts the number of edges we have added/removed from the graph from the current fundamental cycle.
FCCount = 0             #This counts how many fundamental cycles we have used.

treecounter = 0
localcount = 0

frame = 1                 #If you would like to slow down the animation, where you see "  if frame%1 == 0: ", set %1 to %3 or 4. Useful for smaller graphs.
ga = g.copy()               #These are copies of our graphs from above.
ma = m.copy()
ifoa = ifo.copy()
step = 0.1                  #How much a node can move every step of the animation.
K = 5                   #This is the variable which defines the prefered edge length in the animation. This is why the edges all end up a uniform size at the end of the visualisation

edgestoremove = []      #Edges which will be added/removed in our induced forest.
edgestoadd = []

def updateOne():
    global totalcounter
    global cyclecounter
    global FCCount
    global ga
    global frame

    if totalcounter >= len(FCEdges):                    #If we have already XORed all the edges of all the fundamental cycles, we can stop.
        sfdp_layout(ga, pos=pos,K=K,init_step=step,max_iter=1)
        win.graph.regenerate_surface()
        win.graph.queue_draw()                          #These arguments recalculate and then redraw the layout of the graph.
        return True

    Cycle = FundamentalCycles1[FCCount]
    if cyclecounter == 0:                       #If cyclecounter == 0 we have started XORing a new fundamental cycle, so colour the edges of this cycle.
        for edge in ga.edges():
            ecolour[edge] = [0.6, 0.6, 0.6, 1]
        for v in ga.vertices():
            vcolour[v] = [0.640625, 0, 0, 0.9]


        for edge in ga.edges():
            flag = 0

            x = int(edge.source())
            y = int(edge.target())
            for cycleedge in Cycle.edges():                         #This colours edges of the graph depending if they are in the cycle or not.
                xc = int(cycleedge.source())
                yc = int(cycleedge.target())
                if (((xc==x)&(yc==y))|((xc==y)&(yc==x))):
                    flag = 1
            if flag == 1:
                ecolour[edge] = orange
                vcolour[ga.vertex(x)] = blue
                vcolour[ga.vertex(y)] = blue


        sfdp_layout(ga, pos=pos,K=K,init_step=step,max_iter=1)
        win.graph.fit_to_window()
        win.graph.regenerate_surface()
        win.graph.queue_draw()                              #These arguments recalculate and then redraw the layout of the graph.



    if frame%1 == 0:
        if cyclecounter < Cycle.num_edges():                #If we haven't yet XORed every edge in the fundamental cycle, do so.
            ga = XOREdge(ga,FCEdges[totalcounter])
            totalcounter += 1
            cyclecounter += 1

        else:                           #If we have, select a new fundamental cycle.
            cyclecounter = 0
            FCCount += 1

        win.graph.regenerate_surface()                  #Redraw the graph with the same layout as before, making sure to fit it to window.
        win.graph.queue_draw()

    frame += 1

    sfdp_layout(ga, pos=pos,K=K,init_step=step,max_iter=1)
    win.graph.fit_to_window()


    return True         #Must return true for the animation to continue.

def updateTwo():
    global frame
    global totalcounter
    global edgestoremove
    global localcount
    global g



    if totalcounter == 0:   #This will colour any edges to be removed red. This will colour any edges to be added yellow. It will then remove the red edges and add the yellow edges.

        for edge in ma.edges():             #Gives every edge and vertex the default colours.
            x = int(edge.source())
            y = int(edge.target())
            flag = 0
            for edgei in ifoa.edges():              #If an edge is in our odd forest and not our induced off forest, it must be removed. We will colour these edges red.
                xg = int(edgei.source())
                yg = int(edgei.target())
                if (((xg == x)&(yg==y))|((xg==y)&(yg==x))):
                    flag = 1
            if flag == 0:
                ecolour[edge] = [0.640625, 0, 0, 0.9]
                edgestoremove.append(edge)
            else:
                ecolour[edge] = [0.6, 0.6, 0.6, 1]

        for edge in ifoa.edges():           #If an edge is in the induced forest (ifoa), and not in M, it must be added. We will add it to M and colour this edge yellow.
            x = int(edge.source())
            y = int(edge.target())
            flag = 0
            for edgem in ma.edges():
                xm = int(edgem.source())
                ym = int(edgem.target())
                if (((xm==x)&(ym==y))|((xm==y)&(ym==x))):
                    flag = 1
            if flag == 0:
                ma.add_edge(x,y)
                e = ma.edge(x,y)
                ecolour[e] = yellow
                edgestoadd.append(edge)



        print "Number of uninduced edges = " + str(len(edgestoremove))
        if len(edgestoremove) == 0:
            print "This forest was already induced."
            return False

    elif totalcounter == 1:
        user_input = input("Paused. Yellow edges represent uninduced edges which will be added. Red edges represent edges which will be removed. Please enter '1' to continue. ")


    if localcount >= len(edgestoremove):                #We remove the edges from edgestoremove and update the layout.
        totalcounter +=1
        sfdp_layout(ma, pos=pos,K=K,init_step=step,max_iter=1)              #These arguments recalculate and then redraw the layout of the graph.
        win.graph.regenerate_surface()
        win.graph.queue_draw()
        return True


    if totalcounter > 1:
        e = edgestoremove[localcount]
        ma.remove_edge(e)
        localcount += 1



    win.graph.regenerate_surface()
    win.graph.queue_draw()
    sfdp_layout(ma, pos=pos,K=K,init_step=step,max_iter=1)
    win.graph.fit_to_window()

    totalcounter += 1
    return True

def updateThree():
    global totalcounter
    global cyclecounter
    global FCCount
    global ifoa                                                     #This is essentially identical to updateOne, we show a set of fundamental cycles being XORed to the independent forest until it's a perfect matching.
    global frame

    if totalcounter >= len(FCEdges2):                               #If we have already XORed all the edges of all the fundamental cycles, we can stop.
        sfdp_layout(ifoa, pos=pos,K=K,init_step=step,max_iter=1)            #This recalculates the layout of the graph
        win.graph.regenerate_surface()                                      #This redraws the layout of the graph
        win.graph.queue_draw()                                      #This draws the graph again
        for edge in ifoa.edges():
            ecolour[edge] = [0.6, 0.6, 0.6, 1]
        for v in ifoa.vertices():
            vcolour[v] = [0.640625, 0, 0, 0.9]
        return True

    Cycle = FundamentalCycles2[FCCount]
    if cyclecounter == 0:                   #If cyclecounter == 0 we have started XORing a new fundamental cycle, so colour the edges of this cycle.
        for edge in ifoa.edges():
            ecolour[edge] = [0.6, 0.6, 0.6, 1]
        for v in ifoa.vertices():
            vcolour[v] = [0.640625, 0, 0, 0.9]


        for edge in ifoa.edges():
            flag = 0

            x = int(edge.source())
            y = int(edge.target())
            for cycleedge in Cycle.edges():
                xc = int(cycleedge.source())
                yc = int(cycleedge.target())
                if (((xc==x)&(yc==y))|((xc==y)&(yc==x))):
                    flag = 1
            if flag == 1:
                ecolour[edge] = orange
                vcolour[ifoa.vertex(x)] = blue
                vcolour[ifoa.vertex(y)] = blue


        sfdp_layout(ifoa, pos=pos,K=K,init_step=step,max_iter=1)
        win.graph.fit_to_window()
        win.graph.regenerate_surface()
        win.graph.queue_draw()                              #This recalculates and then redraws the layout of the graph


    if frame%1 == 0:
        if cyclecounter < Cycle.num_edges():             #If we haven't yet XORed every edge in the fundamental cycle, do so.
            ifoa = XOREdge(ifoa,FCEdges2[totalcounter])
            totalcounter += 1
            cyclecounter += 1

        else:
            cyclecounter = 0            #If we have, select a new fundamental cycle.
            FCCount += 1

        win.graph.regenerate_surface()              #Redraw the graph with the same layout as before, making sure to fit it to window.
        win.graph.queue_draw()

    frame += 1

    sfdp_layout(ifoa, pos=pos,K=K,init_step=step,max_iter=1)
    win.graph.fit_to_window()


    return True




ecolour = ga.new_edge_property("vector<double>")                                 #This adds a new property to each edge/vertex which gives them a colour. The colour here is just the default one used by GraphTool, but this allows us to update them.
for e in ga.edges():
    ecolour[e] = [0.6, 0.6, 0.6, 1]
vcolour = ga.new_vertex_property("vector<double>")
for v in ga.vertices():
    vcolour[v] = [0.640625, 0, 0, 0.9]

ecolour = ma.new_edge_property("vector<double>")
for e in ma.edges():
    ecolour[e] = [0.6, 0.6, 0.6, 1]
vcolour = ma.new_vertex_property("vector<double>")
for v in ma.vertices():
    vcolour[v] = [0.640625, 0, 0, 0.9]

ecolour = ifoa.new_edge_property("vector<double>")
for e in ifoa.edges():
    ecolour[e] = [0.6, 0.6, 0.6, 1]
vcolour = ifoa.new_vertex_property("vector<double>")
for v in ifoa.vertices():
    vcolour[v] = [0.640625, 0, 0, 0.9]



yellow = [1, 0.8, 0.2, 0.8]                                            #These are three different colours that we will use to highlight different things in the animations.
orange = [0.807843137254902, 0.3607843137254902, 0.0, 1.0]
blue = [0.2, 0.8, 0.8, 0.2]



print("Please enter '1' to see pictures of the graph at different stages of the algorithm. ")
print("Please enter '2' to see an animation of a set of fundamental cycles being XORed to a 3-regular bipartite graph.")
print("Please enter '3' to see how an uninduced forest is turned into an induced forest. ")
print("Please enter '4' to see an animation of a set of fundamental cycles being XORed to an induced forest, turning it into a perfect matching. ")
print ""
argument = input("Please enter a number: ")
print ""

if argument == 1:                                                   #These just draw the graphs into an interactive_window
    print("The input graph g, a 3-regular bipartite graph.")
    graph_tool.draw.interactive_window(g)
    print("An odd forest of g.")
    graph_tool.draw.interactive_window(m)
    print("An induced odd forest, M, of g.")
    graph_tool.draw.interactive_window(ifo)
    print("The complement of M with respects to g.")
    graph_tool.draw.interactive_window(Comp)
    print("The 2-3 graph H given by H = M U N where N is a matching in the complement.")
    graph_tool.draw.interactive_window(twoThreeGraph)
    print("The perfect matching found by the algorithm")
    graph_tool.draw.interactive_window(pm)

if argument == 2:
    print ""
    print("Blue vertices represent vertices in the fundamental cycle.")
    print("Orange edges are edges being added/removed as we XOR the fundamental cycle.")
    cid = GObject.idle_add(updateOne)               #Calls updateOne and updates the animation once finished.
    pos = sfdp_layout(ga, K=K)                      #This gives the initial positions of the vertex.
    win = GraphWindow(ga, pos,edge_color=ecolour,vertex_fill_color=vcolour, geometry=(800, 600))  #Creates the window the graph will be displayed in.
    win.connect("delete_event", Gtk.main_quit)          #Allows us to close the animation by pressing exit on the window.
    win.show_all()                      #Shows and starts the animation.
    Gtk.main()

if argument == 3:
    cid = GObject.idle_add(updateTwo)
    pos = sfdp_layout(ma, K=K)
    win = GraphWindow(ma, pos,edge_color=ecolour,vertex_fill_color=vcolour, geometry=(800, 600))
    win.connect("delete_event", Gtk.main_quit)          #Allows us to close the animation by pressing exit on the window.
    win.show_all()                      #Shows and starts the animation.
    Gtk.main()

if argument == 4:
    print ""
    print("Blue vertices represent vertices in the fundamental cycle.")
    print("Orange edges are edges being added/removed as we XOR the fundamental cycle.")
    cid = GObject.idle_add(updateThree)
    pos = sfdp_layout(ifoa, K=K)
    win = GraphWindow(ifoa, pos,edge_color=ecolour,vertex_fill_color=vcolour, geometry=(800, 600))
    win.connect("delete_event", Gtk.main_quit)          #Allows us to close the animation by pressing exit on the window.
    win.show_all()                      #Shows and starts the animation.
    Gtk.main()
