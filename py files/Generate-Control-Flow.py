#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import copy


# In[2]:


def print_code(a):
    for idx, i in enumerate(a):
        print(i, idx)
    print("")


# In[3]:


def get_ws(line):
    c = 0
    for i in line:
        if i in ' ':
            c+=1
        else:
            return c


# In[ ]:





# In[ ]:





# In[62]:


def find_edges_for(states, end):
    edges = []
    
    edges.append((hash(str(states[0])), hash(str(states[1][0]))))
    edges.append((hash(str(states[1][-1])), hash(str(states[0]))))
    edges.append((hash(str(states[0])), hash(str(end))))

    return edges


# In[63]:


def get_for(lines):
    print("")
    print_code(lines)

    main_stack = []
    stack = []
    
    ws = get_ws(lines[0])

    index = 0
    edges = []
    
    main_stack.append(lines[index])

    index += 1

    while True:
        print(len(lines)-1, index)

        if len(lines)-1 == index:
            print("\n\nreached end")
            if len(stack) > 0:
                print("appended stack")
                main_stack.append(stack)
                print("Stack Now", stack)
                print("Main Stack Now", main_stack)
            end = ""
            print("ends", index)
            return main_stack,  find_edges_for(main_stack, end) + edges, index
        
        try:
            ws_inside = get_ws(lines[index])
        except:
            if len(stack) > 0:
                print("appended stack")
                main_stack.append(stack)
                print("Stack Now", stack)
                print("Main Stack Now", main_stack)
            end = ""
            return main_stack, find_edges_for(main_stack, end) + edges, index
        
        if ws_inside == ws:
            print("armagedon")
            if len(stack) > 0:
                print("appended stack2")
                main_stack.append(stack)
                print("Stack Now2", stack)
                print("Main Stack Now2", main_stack)
            end = ""
#             if get_ws(lines[index]) == ws_inside:
            try:
                end = lines[index]
            except:
                pass
            return main_stack, find_edges_for(main_stack, end) + edges, index

        if ws_inside is None:
            print("\n\nreached end D")
            if len(stack) > 0:
                main_stack.append(stack)
            print("Main Stack Now", main_stack)
            end = ""
            print("*********Exiting For D")
            return main_stack,  find_edges_for(main_stack, end) + edges, index
        
        if ws_inside > ws:
            print("sending to general")
            print(lines[index:])
            states, edges_, length = get_general(lines[index:])
            edges += edges_
            index += length
            stack += states
        
        print(index)
        
        index += 1

    print("\nReturning at last, recheck")
    return main_stack, [], 0


# In[64]:


def get_if(lines):
    print_code(lines)
    lines.append("\n")

    main_stack = []
    
    ws = get_ws(lines[0])
    print(ws)

    if_heads = []
    if_s = []
    
    else_reached = False
    
    p = []
    
    c = 0
    for i in lines:
        ws_i = get_ws(i)
        
        if ws_i is None:
            break
        
        if ws_i == ws:
            if not(i[ws_i:].startswith("if") or i[ws_i:].startswith("elif") or i[ws_i:].startswith("else")):
                if_s.append(p)
                break
            
            if_heads.append(i)
            if len(p) > 0:
                if_s.append(p)
                p = []
        
        if ws_i > ws:
            p.append(i)
        
        c+=1
    if len(p) > 0:
        if_s.append(p)
    print(if_heads)
    print(if_s)
    
    states = []
    edges = []
    for i, j in zip(if_heads, if_s):
        states.append(i)
        states_, edges_, _ = get_general(j)
        states.append(states_)
        edges += edges_
    
    for idx, i in enumerate(states):
        if idx%2 == 0:
            try:
                edges.append((hash(str(states[idx])), hash(str(states[idx+1]))))
            except:
                pass
            try:
                edges.append((hash(str(states[idx])), hash(str(states[idx+2]))))
            except:
                pass
        else:
            try:
                edges.append((hash(str(states[idx])), hash(str(end))))
            except:
                pass
    return states, edges, c-1


# # Processing General

# In[65]:


def find_edges_general(states, end):
    edges = []
    
    edges.append((hash(str(states)), hash(str(states[0]))))
    
    for idx, i in enumerate(states):
        try:
            edges.append((hash(str(states[idx])), hash(str(states[idx+1]))))
        except:
            pass
    try:
        edges.append((hash(str(states[-1])), hash(str(end))))
    except:
        pass
    return edges


# In[91]:


def get_general(lines):
    print("********** Entered General")
    print_code(lines)

    main_stack = []
    stack = []

    ws = get_ws(lines[0])
    print("bit", ws)

    index = 0
    edges = []

    while True:
        print(len(lines), index)
        
        if len(lines) == index:
            print("\n\nreached end")
            if len(stack) > 0:
                print("appended stack")
                main_stack.append(stack)
                print("Stack Now", stack)
                print("Main Stack Now", main_stack)
            end = ""
            print("*********Exiting Genreal C End")
            return main_stack, find_edges_general(main_stack, end) + edges, index
        
        ws_inside = get_ws(lines[index])

        if ws == ws_inside:
            print(lines[index], "appended")
            try:
                if(lines[index][ws : ws+2] == "if" and get_ws(lines[index])>ws):
                    print("if found")
                    print(lines[index:])
                    states, edges_, length = get_if(lines[index:])
                    index += length - 1
                    print(lines[index])
                    edges += edges_
                    print("\n\nappended states")
                    main_stack.append(states)
                elif (lines[index][ws:].startswith("for") or lines[index][ws:].startswith("while")) and get_ws(lines[index])>ws:
                    print("for/while found")
                    print(lines[index:])
                    print("EED", index, len(lines))
                    if index == len(lines)-1:
                        main_stack.append(stack)
                        main_stack.append(lines[index])
                        end = ""
                        print("*********Exiting Genreal EED")
                        return main_stack, find_edges_general(main_stack, end) + edges, index
                    states, edges_, length = get_for(lines[index:])
                    index += length - 1
                    edges += edges_
                    print("\n\nappended states")
                    main_stack.append(states)
                    print("Test", index, len(lines))
                    if index == len(lines):
                        end = ""
                        print("*********Exiting Genreal ED")
                        return main_stack, find_edges_general(main_stack, end) + edges, index
                else:
                    main_stack.append(lines[index])
            except:
                main_stack.append(lines[index])
        if ws_inside is None:
            print("\n\nreached end D")
            if len(stack) > 0:
                main_stack.append(stack)
            print("Main Stack Now", main_stack)
            end = ""
            print("*********Exiting Genreal D")
            return main_stack, find_edges_general(main_stack, end) + edges, index
        
        if ws_inside > ws:
            states, edges_, length = get_general(lines[index:])
            index += length
            edges += edges_
            print("\n\nappended states")
            main_stack.append(states)
            print(main_stack)
            print("Holla", length, index, len(lines))
            if len(lines) == index:
                print("\n\nreached end B")
                if len(stack) > 0:
                    main_stack.append(stack)
                print("Main Stack Now", main_stack)
                end = ""
                print("*********Exiting Genreal A")
                return main_stack, find_edges_general(main_stack, end) + edges, index

        if ws_inside < ws:
            end = ""
            if len(stack) > 0:
                main_stack.append(stack)
                stack = []
                if get_ws(lines[index]) == ws:
                    end = lines[index]
                print("\nReturned correct, pointed to:", end)
            print("*********Exiting Genreal B")
            return main_stack, find_edges_general(main_stack, end) + edges, index-1

        index += 1

    print("\nReturning at last general, recheck")


# In[92]:


# with open("control_flow_test.txt") as f:
#     c = f.read()
#     c = c.split("\n")

# states, edges, length = get_for(c)
# print('')
# print(length)
# print(edges)
# print(states)


# In[93]:


# with open("control_flow_test.txt") as f:
#     c = f.read()
#     c = c.split("\n")

# get_if(c)


# In[94]:


with open("control_flow_test.txt") as f:
    c = f.read()
    c = c.split("\n")

states, edges, length = get_general(c)
print('')
print(length)
# print(edges)
print(states)
for i in edges:
    print(i)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




