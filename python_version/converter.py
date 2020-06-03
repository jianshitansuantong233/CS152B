import nnef
import os, fnmatch
import argparse
import struct
from bitarray import bitarray
import numpy as np
import string
import copy
import torch
import torchvision
import torchvision.transforms as transforms
'''
Compiler for 3PXNet. Compiles a neural network stored in NNEF format to C using inference engine.
'''
'''
NOTIFICATIONS:
   variablen is a dictionary, its keys are variable_# and value is a list, [non-pruned inputs, the operation object whose output name is variable_#]
   variables is also a dictionary, its keys are variable_# and value is the file name of this data
   batchn is also a dictionary, its keys are indices of graph object, where there is a batchnorm operation, the values are variable_#, who are input parameter to this operation 
   The workflow is like this: given a graph, first this program will read all of its operations and determine whether a given operation is able to be compiled or not.
   Then it reads the data files and put the values into header files, i.e., decoding_data. After that, threshold and sign needed for some batchnorm layers are computed. 
   Then it starts writing source file. total_ops stores indices of graph where matrix multiplication, whether conv or fc, takes place.
   To decide whether there is a batchnorm to one layer or not, it look ahead for another matrix multiplication, if there is a batchnorm operation between these two, then there will be a 
   batchnorm, and vice versa.
'''

# WJR: What your code prints out needs to be more descriptive. I would like it to look something like this (not necessarily 100% exact):

#Layer X Output Shape: MxNxLxO
#Translating Layer X: Finished.
#---------------------------------------------
#Translating Layer X+1: Start
#NNEF Layer Primitive: XXXXX
#WARNING: Layer primitive XXXXX not supported. Will be ignored in translation.
#Layer X+1 Output Shape: MxNxLxO
#Translating Layer X+1: Finished.
#---------------------------------------------
#Translating Layer X+2: Start
#NNEF Layer Primitive: YYYYY
#Layer recognized: sparse/dense conv/fc layer with batch norm/pooling/padding
#Layer source files: <list the files containing all parameters for the layer>
#Layer parameters: <lists all the sizes>
#Library C function chosen: <name of the C function used to implement it>
#Output header files: <list all C headers created for this layer>
#Layer X+2 Output Shape: MxNxLxO
#Translating Layer X+2: Finished
#---------------------------------------------
#Translating Layer X+3: Start

# FZ: I tried to print out as much as I can. The C function chosen cannot be printed together with other information
# because they are done in different places. In order to decrease confusion, I decided not to write it out.




class convert(object):
   def __init__(self,input_dir,dataset,test_start_id,test_end_id):
      '''
      initialize a convert object

      :param input_dir: the input directory, its name should end with .nnef
      :param dataset: dataset to test against
      '''
      self.input_dir=input_dir
      self.dataset=dataset
      self.test_start_id=int(test_start_id)
      self.test_end_id=int(test_end_id)
      self.graph=nnef.Graph
      #batch norm variables
      self.var = {}
      self.mean = {}
      self.gamma = {}
      self.beta = {}
      # store variables with their names as keys
      # for specific information, please see NOTIFICATIONS above
      self.variablen = {}
      self.variables = {}
      # store index of propagation in the graph
      self.matrixmul = []
      self.conv = []
      self.batchn = {}
      #input shape
      self.in_shape = []
      self.rank = []
      # which batch norm layer is the last one
      self.batch_last=" "
      # source code we are writing to
      self.source=0
      #permutation list. For specific information, please see training engine
      #as well as the paper the whole project is based on.
      self.list=[]
      self.tempweight=[]
      self.tempsparse=False
      self.tempoutput=0
      self.name=" "
      self.lastlist=[]
      self.tempvar=" "
      self.tempmean=""
      self.tempgamma=""
      self.tempbeta=""

   def loadgraph(self):
      '''
      load the nnef graph into compiler
      '''
      if "autogen" not in os.listdir("."):
         os.mkdir("autogen")
      print(self.input_dir)
      os.chdir(self.input_dir)
      if "graph.nnef" not in os.listdir("."):
         print("ERROR: BAD NNEF DIRECTORY!")
         exit()
      else:
         self.graph = nnef.load_graph('graph.nnef')
      print("Graph loaded")

   def writebias(self,rank,temp_array,name,output):
      indices = []
      print("Writing to header " + name + ".h ...")
      output.write("#define " + name + " {\\\n")
      # NNEF format weight values are stored in row-major order.
      # So for a fc layer, its shape is [input, output]
      for i in range(rank):
         # outtemp is used to store packs
         # mask is used to check whether a given pack is all zero
         temp = temp_array[i]
         output.write(str(temp) + ", \\\n")
      output.write("}\n")
      output.close()
   def writefc(self,write,rank,temp_array,sparse,output,name):
      '''
      write fc layer's data into C headers

      :param write: whether to write or not(related to permutation issue)
      :param rank: weight's shape
      :param temp_array: weight data
      :param sparse: whether the layer is sparse or not
      :param output: IO object, corresponding to the header file it's writing to
      :param name: name of the header file
      :return: indices: if it is a sparse layer, indices are used to calculate # non-pruned inputs
      '''
      indices=[]
      if write:
         print("Writing to header "+name+".h ...")
         output.write("#define " + name + " {\\\n")
      #NNEF format weight values are stored in row-major order.
      #So for a fc layer, its shape is [input, output]
      for i in range(rank[0]):
         #outtemp is used to store packs
         #mask is used to check whether a given pack is all zero
         for j in range(rank[1]):
            temp = temp_array[i, j]
            output.write(str(temp) + ", \\\n")
      if write:
         output.write("}\n")
         if sparse:
            output.write("#define " + name + "_indices {\\\n")
            for i in range(len(indices)):
               output.write(str(indices[i]) + ", \\\n")
            output.write("}\n")
         output.close()
      return indices


   def writecn(self,write,rank,temp_array,sparse,output,name):
      '''
      write conv layer's data into C headers
      The same as fc layer, NNEF format stores value in row-major order
      So for a conv layer, the shape is [n,z,y,x]
      But, I modified this order during decoding data time.
      So now the input rank has a shape [x,y,z,n]

      :param write: whether to write or not(related to permutation issue)
      :param rank: weight's shape
      :param temp_array: weight data
      :param sparse: whether the layer is sparse or not
      :param output: IO object, corresponding to the header file it's writing to
      :param name: name of the header file
      :return: indices: if it is a sparse layer, indices are used to calculate # non-pruned inputs
      '''
      indices=[]
      if write:
         print("Writing to header "+name+'.h ...')
         output.write("#define " + name + " {\\\n")
      # WJR: Why is there a second docstring here. Does python even support that?
      # FZ: fixed
      for n in range(rank[0]):
         #outtemp is used to store packs
         #mask is used to check whether a given pack is all zero
         for y in range(rank[2]):
            for x in range(rank[3]):
               for z in range(rank[1]):
                  temp = temp_array[x, y, z, n]
                  output.write(str(temp) + ", \\\n")
                  indices.append(int(z / 32) + x * int(rank[1] / 32) + y * rank[3] * int(
                              rank[1] / 32))
      if write:
         output.write("}\n")
         if sparse:
            output.write("#define " + name + "_indices {\\\n")
            for i in range(len(indices)):
               output.write(str(indices[i]) + ", \\\n")
            output.write("}\n")
         output.close()
      return indices


   # WJR: change the name of this function. "Processing data" means absolutely nothing.
   # What data are you processing? How are you processing it?
   # Same with the docstring below: "processing" means nothing and everything.
   # Please be more specific.
   # FZ: fixed
   def decoding_data(self, input, output, name, last, identity,first):
      '''
      processing a given .dat file stored in NNEF format
      to be specific, a NNEF formtted neural network contains a .graph file and several .dat files.
      this function deals with a given .dat file. it first reads in specifications of this file, such as
      its length and its shape. Then it will translate weights stored in binary in this .dat file into packs or digits.
      The actual writing-to-header process is done by writecn and writefc functions.

      :param input: IO object, corresponding to the .dat file it's reading from
      :param output: IO object, corresponding to the header file it's writing to
      :param name: name of the header file
      :param last: whether the given .dat file is corresponded with the last batch norm layer
      :param identity: whether the given .dat file contains values for conv/fc/batchnorm layer
      :param first: whether it is the first matrix operations in the graph. Needed for permutation issue
      :return: if the input file is a fc or cn layer and it's sparse, then non-pruned inputs number is returned.
            if the input file is a batch norm layer and it's not the last one, a list with all its values are returned.
            otherwise, return 0
      '''
      input.read(4)
      length = int.from_bytes(input.read(4), byteorder='little')
      rank_n = int.from_bytes(input.read(4), byteorder='little')
      rank = []  # n,z,y,x
      batch = (identity == 0)
      fc = (identity == 1)
      cn = (identity == 2)
      bias=(identity==3)
      for i in range(0, rank_n):
         rank.append(int.from_bytes(input.read(4), byteorder='little'))
      input.read((8 - rank_n) * 4)
      bits_per_item = int.from_bytes(input.read(4), byteorder='little')
      input.read(2)
      size = int(bits_per_item / 8)
      # interpret as float or int
      algo = int.from_bytes(input.read(2), byteorder='big')
      signess = int.from_bytes(input.read(4), byteorder='little')
      # TODO: more about linear and log quantize later
      # reference: https://www.khronos.org/registry/NNEF/specs/1.0/nnef-1.0.2.html#container-structure
      input.seek(128, 0)
      # start reading data
      sparse = False
      indices = []
      result = []
      # fc needs to be packed in column-major order
      if bias:
         if (len(rank)==2):
            rank=rank[1]
         else:
            rank=rank[0]
         temp_array=np.zeros(rank)
         for i in range(rank):
            temp = list(input.read(size))
            # changing endianess
            for b in range(0, int(len(temp) / 2)):
               temp1 = temp[b]
               temp[b] = temp[len(temp) - b - 1]
               temp[len(temp) - b - 1] = temp1
            temp = bytes(temp)
            # decode as float
            if struct.unpack('!f', temp)[0] == 0:
               sparse = True
            temp_array[i] = struct.unpack('!f', temp)[0]
         self.writebias(rank,temp_array,name,output)
      elif fc:
         temp_array = np.zeros((rank[0], rank[1]))
         for i in range(rank[0]):
            for j in range(rank[1]):
               temp = list(input.read(size))
               #changing endianess
               for b in range(0, int(len(temp) / 2)):
                  temp1 = temp[b]
                  temp[b] = temp[len(temp) - b - 1]
                  temp[len(temp) - b - 1] = temp1
               temp = bytes(temp)
               #decode as float
               if struct.unpack('!f', temp)[0] == 0:
                  sparse = True
               temp_array[i, j] = struct.unpack('!f', temp)[0]
         # permutation
         os.chdir('..')
         os.chdir(self.input_dir)
         flag=False
         for root, dirs, files in os.walk("."):
            for name1 in files:
               if fnmatch.fnmatch(name1.replace('_', ''), name.replace('weight', 'list.npy').replace('_', '')):
                  print("Permuting...")
                  flag=True
                  temp_weight = np.zeros((rank[0], rank[1]))
                  permute_list = np.load(name1)
                  if first:
                     self.list=permute_list
                  #permute input channel for current layer so that we can pack weights
                  for i in range(rank[0]):
                     temp_weight[i, 0:] = np.copy(temp_array[permute_list[0, i], 0:])
                  #permute output channel for last layer so that channels match
                  if len(self.tempweight)!=0:
                     tt=np.copy(self.tempweight)
                     if len(tt.shape)==4:
                        for j in range(tt.shape[3]):
                           self.tempweight[0:, 0:, 0:, j] = np.copy(tt[0:, 0:, 0:,permute_list[0, j]])
                        self.writecn(True,[tt.shape[3],tt.shape[2],tt.shape[1],tt.shape[0]],self.tempweight,
                                     self.tempsparse,self.tempoutput,self.name)
                     else:
                        for i in range(rank[0]):
                           self.tempweight[0:, i] = np.copy(tt[0:, permute_list[0, i]])
                        self.writefc(True,tt.shape,self.tempweight,self.tempsparse,self.tempoutput,self.name)
                     #permute the last batch layer as well
                     tt=np.copy(self.var[self.tempvar])
                     for i in range(rank[0]):
                        self.var[self.tempvar][i] = np.copy(tt[permute_list[0, i]])
                     tt = np.copy(self.mean[self.tempmean])
                     for i in range(rank[0]):
                        self.mean[self.tempmean][i] = np.copy(tt[permute_list[0, i]])
                     tt = np.copy(self.gamma[self.tempgamma])
                     for i in range(rank[0]):
                        self.gamma[self.tempgamma][i] = np.copy(tt[permute_list[0, i]])
                     tt = np.copy(self.beta[self.tempbeta])
                     for i in range(rank[0]):
                        self.beta[self.tempbeta][i] = np.copy(tt[permute_list[0, i]])
                  temp_array = temp_weight
                  #save this layer's state so that later we can permute its output channel
                  self.tempweight=temp_array
                  self.tempoutput=output
                  self.tempsparse=sparse
                  self.name=name
                  self.lastlist=permute_list
                  break
         #if there is nothing to be permuted, meaning this layer is not on the temp state in this class
         # so we directly write them into header file
         # otherwise, wait for it to be permuted by next layer
         if flag:
            indices=self.writefc(False,rank,temp_array,sparse,output,name)
         else:
            indices = self.writefc(True, rank, temp_array, sparse, output, name)
         os.chdir("..")
         os.chdir("autogen")

      elif cn:
         # first layer in a cnn
         # it uses binarized dense layer, so we don't pack it
         if rank[1] % 32 != 0:
            output.write("#define " + name + " {\\\n")
            temp_array = np.zeros((rank[0], rank[1], rank[2], rank[3]))
            for n in range(rank[0]):
               for z in range(rank[1]):
                  for y in range(rank[2]):
                     for x in range(rank[3]):
                        temp = list(input.read(size))
                        #changing endianess
                        for b in range(0, int(len(temp) / 2)):
                           temp1 = temp[b]
                           temp[b] = temp[len(temp) - b - 1]
                           temp[len(temp) - b - 1] = temp1
                        temp = bytes(temp)
                        if struct.unpack('!f', temp)[0] == 0:
                           sparse = True
                        temp_array[n, z, y, x] = struct.unpack('!f', temp)[0]
            print("Sparse?: " + str(sparse))
            for n in range(rank[0]):
               for y in range(rank[2]):
                  for x in range(rank[3]):
                     for z in range(rank[1]):
                        temp = temp_array[n, z, y, x]
                        output.write(str(temp) + ", ")
                  output.write('\\\n')
            output.write("}\n")
            output.close()
         # other conv layers in a cnn
         else:
            temp_array = np.zeros((rank[3], rank[2], rank[1], rank[0]))
            for n in range(rank[0]):
               for z in range(rank[1]):
                  for y in range(rank[2]):
                     for x in range(rank[3]):
                        temp = list(input.read(size))
                        #changing endianess
                        for b in range(0, int(len(temp) / 2)):
                           temp1 = temp[b]
                           temp[b] = temp[len(temp) - b - 1]
                           temp[len(temp) - b - 1] = temp1
                        temp = bytes(temp)
                        if struct.unpack('!f', temp)[0] == 0:
                           sparse = True
                        temp_array[x, y, z, n] = struct.unpack('!f', temp)[0]
            print("Sparse?: "+str(sparse))
            # permutation
            os.chdir('..')
            os.chdir(self.input_dir)
            flag=False
            for root, dirs, files in os.walk("."):
               for name1 in files:
                  if fnmatch.fnmatch(name1.replace('_', ''), name.replace('weight', 'list.npy').replace('_', '')):
                     print("Permuting...")
                     flag=True
                     temp_weight = np.zeros((rank[3], rank[2], rank[1], rank[0]))
                     permute_list = np.load(name1)
                     if first:
                        self.list=permute_list
                     #permute input channel of current layer
                     for j in range(rank[0]):
                        for i in range(rank[1]):
                           temp_weight[0:, 0:, i, j] = np.copy(temp_array[0:, 0:, permute_list[0, i], j])
                     #permute output channel of last layer
                     #since it's not possible to have a fc layer before a conv layer,
                     #we don't consider that case here.
                     if len(self.tempweight)!=0:
                        tt=np.copy(self.tempweight)
                        for j in range(tt.shape[3]):
                           self.tempweight[0:, 0:, 0:, j] = np.copy(tt[0:, 0:, 0:,permute_list[0, j]])
                        self.writecn(True, [tt.shape[3], tt.shape[2], tt.shape[1], tt.shape[0]],
                                     self.tempweight, self.tempsparse, self.tempoutput,self.name)
                        # permute the last batch layer as well
                        tt = np.copy(self.var[self.tempvar])
                        for i in range(rank[0]):
                           self.var[self.tempvar][i] = np.copy(tt[permute_list[0, i]])
                        tt = np.copy(self.mean[self.tempmean])
                        for i in range(rank[0]):
                           self.mean[self.tempmean][i] = np.copy(tt[permute_list[0, i]])
                        tt = np.copy(self.gamma[self.tempgamma])
                        for i in range(rank[0]):
                           self.gamma[self.tempgamma][i] = np.copy(tt[permute_list[0, i]])
                        tt = np.copy(self.beta[self.tempbeta])
                        for i in range(rank[0]):
                           self.beta[self.tempbeta][i] = np.copy(tt[permute_list[0, i]])
                     temp_array = temp_weight
                     #save this layer's state so that later we can permute its output channel
                     self.tempweight=temp_array
                     self.tempoutput=output
                     self.tempsparse=sparse
                     self.name=name
                     self.lastlist=permute_list
                     break
            if flag:
               indices=self.writecn(False,rank,temp_array,sparse,output,name)
            else:
               indices = self.writecn(True, rank, temp_array, sparse, output, name)
            os.chdir("..")
            os.chdir("autogen")

      # batchnorm
      else:
         print("Writing to header "+name+".h ...")
         for i in range(int(length / size)):
            # WJR: explain the comment below please. Mark it as a "TODO"
            # Whenever you make a comment about something to be done later, start it with a TODO
            # Then you can quickly search for them
            # FZ: fixed
            # One great feature of NNEF is it doesn't use many concrete data types. Therefore, there are several
            # encoding algorithms provided. Since current training engine will not train weights whose data types
            # are not float, this converter does not support any other encoding algorithm
            #TODO: depending on encoding algorithm, theoretically we should decode numbers in different ways
            #TODO: more support for this later
            #reference: https://www.khronos.org/registry/NNEF/specs/1.0/nnef-1.0.2.html#container-structure
            if algo == 0:
               temp = list(input.read(size))
               #changing endianess
               for j in range(0, int(len(temp) / 2)):
                  temp1 = temp[j]
                  temp[j] = temp[len(temp) - j - 1]
                  temp[len(temp) - j - 1] = temp1
               temp = bytes(temp)
               if last and "var" in name:
                  #what we really need is standard deviation, not variance
                  # because it will be considered in inference
                  output.write(str(np.sqrt(struct.unpack('!f', temp)[0])) + ", \\\n")
               else:
                  output.write(str(struct.unpack('!f', temp)[0]) + ", \\\n")
            elif algo == 1 and signess == 0:
               output.write(str(int.from_bytes(input.read(size), byteorder='little')) + ", \\\n")
            else:
               output.write(str(int.from_bytes(input.read(size), byteorder='little', signed=True)) + ", \\\n")
         output.write("}\n")
      if batch:
         return result
      # return non-pruned input number
      elif sparse and fc:
         return int(32 * len(indices) / rank[1])
      elif sparse and cn:
         return int(32 * len(indices) / rank[0])
      else:
         return 0


   # WJR: what do you need this for?
   # FZ: currently it replaces '.' in the file name to '_'
   # the reason is if a file name contains a '.', i believe either the compiler or Makefile will give an error
   def replace_non_ascii(self,stri):
      '''
         replace all non ascii, including . , in the file name to _

      :param stri: input string
      :return: the input string with non-character or non-digit being replaced by _
      '''
      return ''.join([i if i in string.ascii_letters or i in string.digits else '_' for i in stri])


   def search_non_ascii(self,stri):
      '''
         search for the first letter that is not letter or digit
         needed for determine last layer of batch norm

      :param stri: input string
      :return: the first index of non-character or non-digit char
      '''
      for i in range(len(stri)):
         if not (stri[i] in string.ascii_letters or stri[i] in string.digits):
            return i


   def find_batch_last(self):
      '''
         the last layer will not be binarized, so batchnorm has to be delt with differently
         determine the last layer

      :return: NA
      '''
      #find out the last matrix multiplication or convolution operation
      batch_last = next(i for i in reversed(range(len(self.graph.operations))) if
                        self.graph.operations[i].name == 'matmul' or self.graph.operations[i].name == 'conv')
      flag=False
      #if there is batch norm after the last matmul or conv, then that is the last batch norm layer
      for i in range(batch_last,len(self.graph.operations)):
         if self.graph.operations[i].name=='batch_normalization':
            for ops in self.graph.operations:
               if ops.outputs['output']==self.graph.operations[i].inputs['mean']:
                  batch_last=ops.attribs['label']
                  flag=True
                  break
      if flag:
         self.batch_last=batch_last[0:self.search_non_ascii(batch_last)]
      else:
         self.batch_last=" "

   def write_source_first(self):
      '''
         write to the source file some include headers

      :return: NA
      '''
      os.chdir("..")
      os.chdir("autogen")
      source = open("source.c", 'w+')
      self.source=source
      source.write("#include <stdio.h>\n")
      source.write("#include <stdlib.h>\n")
      source.write("#include <stdint.h>\n")
      source.write("#include <string.h>\n")
      source.write("#include <math.h>\n")
      source.write("#include <time.h>\n")
      source.write("#include <errno.h>\n")
      source.write("#include \"datatypes.h\"\n")
      source.write("#include \"utils.h\"\n")
      source.write("#include \"xnor_base.h\"\n")
      source.write("#include \"xnor_fc.h\"\n")
      source.write("#include \"xnor_fc.h\"\n")
      source.write("#include \"bwn_dense_cn.h\"\n")
      source.write("#define NULL ((void*)0) \n")
      os.chdir("..")
      os.chdir(self.input_dir)


   def processing_graph(self):
      '''
         for every operation in the graph, determine whether we can translate into C using inference engine or not
         if not, then there will be WARNING or ERROR printed on screen
         if we can, then corresponding data files are decoded and written
         it uses a lot of dictionary type data structures. For detailed information, please see NOTIFICATIONS

      :return: NA
      '''
      rank=[]
      i=0
      for ops in self.graph.operations:
         print("-----------------------------------------")
         print("Operation #"+str(i)+": Start")
         print("Operation name: "+ops.name)
         if ops.name=='avg_pool':
            i+=1
            continue
         elif ops.name == 'mul':
            bias = ops.inputs['y']
            for t in self.graph.operations:
               #find out which file is its data
               if 'output' in t.outputs.keys() and t.outputs['output']==bias:
                  self.variablen[t.outputs['output']]=[]
                  print("Reading weight data from "+t.attribs['label']+".dat ...")
                  ma=open(t.attribs['label']+'.dat','rb')
                  os.chdir("..")
                  os.chdir("autogen")
                  head=open(self.replace_non_ascii(t.attribs['label'])+'.h','w+')
                  self.decoding_data(ma,head,self.replace_non_ascii(t.attribs['label']),False,3, \
                                         len(self.matrixmul)==0 and len(self.conv)==0)
                  self.variablen[t.outputs['output']].append(t)
                  self.source.write("#include \""+self.replace_non_ascii(t.attribs['label'])+'.h'+"\" \n")
                  self.variables[t.outputs['output']]=t.attribs['label']
                  ma.close()
                  os.chdir("..")
                  os.chdir(self.input_dir)
                  break
               elif 'output' not in t.outputs.keys():
                  break;
         elif ops.name =='relu' or ops.name =='add':
            i+=1
            print("WARNING: except for multiplying by 1, mul is not supported")
            continue
         #a convolutional layer
         elif ops.name =='conv':
            #the convolution filter/kernel
            mat=ops.inputs['filter']
            bias=ops.inputs['bias']
            for t in self.graph.operations:
               #find out which file is its data
               if 'output' not in t.outputs.keys():
                  break;
               elif t.outputs['output']==mat:
                  self.variablen[t.outputs['output']]=[]
                  print("Reading weight data from "+t.attribs['label']+".dat ...")
                  ma=open(t.attribs['label']+'.dat','rb')
                  os.chdir("..")
                  os.chdir("autogen")
                  head=open(self.replace_non_ascii(t.attribs['label'])+'.h','w+')
                  npi=self.decoding_data(ma,head,self.replace_non_ascii(t.attribs['label']),False,2, \
                                         len(self.matrixmul)==0 and len(self.conv)==0)
                  if npi!=0:
                     print("Packs per kernel #: "+str(int(npi/32)))
                  self.variablen[t.outputs['output']].append(npi)
                  self.variablen[t.outputs['output']].append(t)
                  self.source.write("#include \""+self.replace_non_ascii(t.attribs['label'])+'.h'+"\" \n")
                  self.variables[t.outputs['output']]=t.attribs['label']
                  ma.close()
                  os.chdir("..")
                  os.chdir(self.input_dir)
                  assert len(rank)==4
                  print("Padding: "+str(ops.attribs['padding'][0][0]))
                  #update current data shape
                  rank[3]=rank[3]+2*ops.attribs['padding'][0][0]-t.attribs['shape'][3]+1
                  rank[2] = rank[2] + 2 * ops.attribs['padding'][0][0] - t.attribs['shape'][2] + 1
                  rank[1] = copy.deepcopy(t.attribs['shape'][0])
                  rank[0]=1
                  if ops.attribs['stride']!=[1,1]:
                     print("ERROR: current 3PXNet does not support stride")
                     exit()
               elif t.outputs['output']==bias:
                  self.variablen[t.outputs['output']] = []
                  print("Reading weight data from " + t.attribs['label'] + ".dat ...")
                  ma = open(t.attribs['label'] + '.dat', 'rb')
                  os.chdir("..")
                  os.chdir("autogen")
                  head = open(self.replace_non_ascii(t.attribs['label']) + '.h', 'w+')
                  self.decoding_data(ma, head, self.replace_non_ascii(t.attribs['label']), False, 3, \
                                     len(self.matrixmul) == 0 and len(self.conv) == 0)
                  self.variablen[t.outputs['output']].append(t)
                  self.source.write("#include \"" + self.replace_non_ascii(t.attribs['label']) + '.h' + "\" \n")
                  self.variables[t.outputs['output']] = t.attribs['label']
                  ma.close()
                  os.chdir("..")
                  os.chdir(self.input_dir)
                  break
            self.conv.append(i)
         #a fc layer
         elif ops.name=='matmul':
            #the kernel of fc layer
            mat=ops.inputs['B']
            for t in self.graph.operations:
               if t.outputs['output']==mat:
                  self.variablen[t.outputs['output']]=[]
                  ma=open(t.attribs['label']+'.dat','rb')
                  os.chdir("..")
                  os.chdir("autogen")
                  head=open(self.replace_non_ascii(t.attribs['label'])+'.h','w+')
                  print("Reading weight data from " + t.attribs['label'] + ".dat ...")
                  npi=self.decoding_data(ma,head,self.replace_non_ascii(t.attribs['label']),False,1,
                                           len(self.matrixmul)==0 and len(self.conv)==0)
                  if npi!=0:
                     print("Packs per kernel #: "+str(int(npi/32)))
                  self.variablen[t.outputs['output']].append(npi)
                  self.variablen[t.outputs['output']].append(t)
                  self.source.write("#include \""+self.replace_non_ascii(t.attribs['label'])+'.h'+"\" \n")
                  self.variables[t.outputs['output']]=t.attribs['label']
                  ma.close()
                  os.chdir("..")
                  os.chdir(self.input_dir)
                  assert len(rank)==2
                  rank[1]=t.attribs['shape'][1]
                  rank[0]=rank[0]
                  break
            self.matrixmul.append(i)
         #externally imported data, currently treated as input
         elif ops.name=='external':
            print("externally imported data, currently treated as input")
            print("WARNING: if it is not used as input, there will be errors.")
            self.in_shape=ops.attribs['shape']
            rank=self.in_shape.copy()
         # batch norm
         elif ops.name=='batch_normalization':
            mat=ops.inputs['mean']
            last=False
            #determine the last batch layer
            last=True
            print("Is the last batch normalization layer: "+str(last))
            #if it is the last batch norm layer, write out everything into the header file
            for b in range(4):
               if b ==0:
                  mat=ops.inputs['mean']
               elif b ==1:
                  mat=ops.inputs['variance']
               elif b ==2:
                  mat=ops.inputs['offset']
               else :
                  mat=ops.inputs['scale']
               for t in self.graph.operations:
                  if t.outputs['output']==mat:
                     assert t.attribs['shape'][0]==rank[0]
                     assert t.attribs['shape'][1] == rank[1]
                     self.variablen[t.outputs['output']]=[]
                     ma=open(t.attribs['label']+'.dat','rb')
                     os.chdir("..")
                     os.chdir("autogen")
                     print("Reading weight data from " + t.attribs['label'] + ".dat ...")
                     head=open(self.replace_non_ascii(t.attribs['label'])+'.h','w+')
                     head.write("#define "+self.replace_non_ascii(t.attribs['label'])+" {\\\n")
                     self.decoding_data(ma,head,self.replace_non_ascii(t.attribs['label']),True,0,False)
                     head.close()
                     self.source.write("#include \""+self.replace_non_ascii(t.attribs['label'])+'.h'+"\" \n")
                     self.variables[t.outputs['output']]=t.attribs['label']
                     ma.close()
                     os.chdir("..")
                     os.chdir(self.input_dir)
                     break
         #if pooling
         elif ops.name=='max_pool':
            rank[3]=int(rank[3]/ops.attribs['size'][3])
            rank[2]=int(rank[2]/ops.attribs['size'][2])
            rank[1] = int(rank[1] / ops.attribs['size'][1])
            rank[0] = int(rank[0] / ops.attribs['size'][0])
         #clamp is considered as binarize output. If it is not used in this way, error would be given
         elif ops.name=='clamp':
            print("WARNING: clamp is considered as binarize output. If it is not used in this way, error would be given")
            if ops.inputs['a']!=-1 or ops.inputs['b']!=1:
               print("ERROR: 3PXNet inference library only holds 1 or -1.")
               exit()
         #current library does not have reshape function. so if the reshaped dimension is not found in the
         #current shape, then error would be given
         elif ops.name=='reshape':
            rank_flag=False
            temp=1
            for r in range(len(rank)):
               temp*=rank[r]
            for r in range(len(ops.attribs['shape'])):
               if temp==ops.attribs['shape'][r]:
                  rank_flag=True
            if not rank_flag:
               print("ERROR: current 3PXNet library does not support reshaping")
               exit()
            rank=copy.deepcopy(ops.attribs['shape'])
            for r in range(len(rank)):
               if rank[r]==-1:
                  rank[r]=1
         #softmax is always added at the end. If not, then the compiler would only give an error
         elif ops.name=='softmax':
            print("WARNING: current 3PXNet library does not support softmax function")
         #these three operations don't have impact on performance
         elif ops.name=='squeeze':
            #for r in ops.attribs['axes']:
            #   del rank[r]
            print("Squeeze has no effect on inference, therefore it is skipped")
         elif ops.name=='unsqueeze' :
            #for r in ops.attribs['axes']:
            #   rank.insert(r,1)
            print("Unsqueeze has no effect on inference, therefore it is skipped")
         elif ops.name=='variable':
            print('Define '+ops.attribs['label']+ ' as '+ops.outputs['output'])
            i+=1
            continue
         #same as softmax, if it's not used in the end, error would be given
         elif ops.name=='log' :
            if ops!=self.graph.operations[-1]:
               print("ERROR: current 3PXNet does not support log function")
               exit()
            else:
               print("WARNING: current 3PXNet does not support log function, but it doesn't affect the result")
         elif ops.name=='slice':
            rank[ops.attribs['axes'][0]]=ops.attribs['end'][0]
            print("WARNING: slice operation is skipped. If this is not operating on the input, the result will be wrong")
         else:
            print("ERROR: current 3PXNet does not support "+ops.name+" function")
            exit()
         print("Operation output shape:",end=' ')
         print(rank)
         i+=1


   def calculate_batch(self):
      '''
         calculate batch normalization threshold and signs

      :return: NA
      '''
      os.chdir("..")
      os.chdir("autogen")
      if len(self.var.keys()) != len(self.mean.keys()) or len(self.var.keys()) != len(self.gamma.keys()) or \
              len(self.var.keys()) != len(self.beta.keys()):
         print("error with batch normalization number")
         exit()
      thresh={}
      sign={}
      k=0
      #calculate threshold and sign
      #variables: map a variable_# to its name stored in nnef directory
      for i in self.batchn.keys():
         if self.variables[self.batchn[i][0]][0:self.search_non_ascii(self.variables[self.batchn[i][0]])] == self.batch_last:
            continue
         temp=bitarray()
         epsilon=[self.graph.operations[i].attribs['epsilon']]*len(self.var[self.batchn[i][1]])
         thresh[k]=[]
         sign[k]=[]
         for j in range(len(self.var[self.batchn[i][1]])):
            thresh[k].append(self.mean[self.batchn[i][0]][j]-np.sqrt(self.var[self.batchn[i][1]][j]+epsilon[j])/
                             self.gamma[self.batchn[i][3]][j]*self.beta[self.batchn[i][2]][j])
         for j in range(len(self.var[self.batchn[i][1]])):
            temp.append(int(self.gamma[self.batchn[i][3]][j]>0))
            if j%32 == 31 :
               sign[k].append(str("0x%x" % int(temp.to01(), 2)))
               temp=bitarray()
         head=open("bn"+str(k+1)+".h",'w+')
         head.write("#define bn"+str(k+1)+"_thresh {\\\n")
         for j in range(len(self.var[self.batchn[i][1]])):
            head.write(str(thresh[k][j])+", \\\n")
         head.write("} \n")
         head.write("#define bn"+str(k+1)+"_sign {\\\n")
         for j in range(int(len(self.var[self.batchn[i][1]])/32)):
            head.write(str(sign[k][j])+", \\\n")
         head.write("} \n")
         self.source.write("#include \"bn"+str(k+1)+".h\" \n")
         k+=1


   # WJR: not sure if I like this mechanism, it doesn't look robust.
   # How does a sparse network nnef differ from a dense one?
   # FZ: yeah i understand it doesn't look nice, but the thing is a sparse network has no difference with a
   # dence one in nnef. For a layer, nnef only stores its values and corresponding specfications in .dat file.
   # The graph just records its shape.
   # Only sparse networks need indices, so I think this method should work fine.
   def testsparse(self,i,total_ops):
      '''
         test sparsity.
         the way to do this is to search for the word "indices" in a given header file

      :param i: index of current operation in "total_ops" list
      :param total_ops: a list of indices of all matrix multiplication/convolution in graph.operations
      :return: whether this layer is sparse or not
      '''
      if 'filter' in self.graph.operations[total_ops[i]].inputs:
         test_sparse = open(
            self.replace_non_ascii(self.variables[self.graph.operations[total_ops[i]].inputs['filter']]) + '.h', 'r')
         sparse = test_sparse.read().find("indices", 0, -1)
      else:
         test_sparse = open(self.replace_non_ascii(self.variables[self.graph.operations[total_ops[i]].inputs['B']]) + '.h',
                            'r')
         sparse = test_sparse.read().find("indices", 0, -1)
      if sparse == -1:
         sparse = False
      else:
         sparse = True
      test_sparse.close()
      return sparse

   # WJR: More descritptive docstring please
   # FZ: fixed
   def write_source_second(self):
      '''
      Write out all remaining source code
      It can be considered as two parts: the first one writes out all specifications of one layer, such as its input
      size, kernel size, and output size. For a convolutional layer, padding and pooling information are also defined
      in this part. Besides, batch normalization information is defined in this part as well.
      The second part writes out used functions defined in inference engine according to different layer settings.

      :return: NA
      '''
      #write source file: first part
      print("-----------------------------------------")
      self.source.write('#include \"image.h\"\n')
      self.source.write("static uint16_t   labels[] = LABELS; \n")
      var={}
      mat_flag=False
      mat_name=" "
      for ops in self.graph.operations:
         if ops.name=='batch_normalization':
            if mat_flag:
               mat_name = str(ops.outputs['output'])
            var[ops.outputs['output']] = copy.deepcopy(var[str(ops.inputs['input'])])
         elif ops.name=='external':
            self.source.write("static float input[] = IMAGES ; \n")
            var['input']=copy.deepcopy(ops.attribs['shape'])
         elif ops.name=='variable':
            self.source.write("static float "+str(ops.outputs['output'])+"[] = "+self.replace_non_ascii(str(ops.attribs['label']))+"; \n")
            var[str(ops.outputs['output'])]=copy.deepcopy(ops.attribs['shape'])
         elif ops.name=='unsqueeze':
            rank = copy.deepcopy(var[str(ops.inputs['input'])])
            length = 1
            for i in range(len(rank)):
               length *= rank[i]
            self.source.write("static float* "+str(ops.outputs['output'])+"="+str(ops.inputs['input'])+"; \n")
            rank=copy.deepcopy(var[str(ops.inputs['input'])])
            for r in ops.attribs['axes']:
               rank.insert(r, 1)
            var[str(ops.outputs['output'])]=copy.deepcopy(rank)
         elif ops.name=='squeeze':
            self.source.write('static float* '+str(ops.outputs['output'])+"="+str(ops.inputs['input'])+"; \n")
            rank=copy.deepcopy(var[str(ops.inputs['input'])])
            for r in ops.attribs['axes']:
               del rank[r]
            var[str(ops.outputs['output'])]=copy.deepcopy(rank)
         elif ops.name=='mul':
            length=1
            for i in range(len(var[str(ops.inputs['y'])])):
               length*=var[str(ops.inputs['y'])][i]
            self.source.write("static float "+str(ops.outputs['z'])+"["+str(length)+"]; \n")
            var[str(ops.outputs['z'])]=copy.deepcopy(var[str(ops.inputs['y'])])
            if mat_flag and str(ops.inputs['y']) == mat_name:
               mat_name = str(ops.outputs['z'])
         elif ops.name=='conv':
            if mat_flag:
               length = 1
               for i in range(len(var[mat_name])):
                  length *= var[mat_name][i]
               self.source.write("static float "+mat_name+"["+str(length)+"]; \n")
            mat_name=str(ops.outputs['output'])
            rank=copy.deepcopy(var[str(ops.inputs['input'])])
            rank[3] = rank[3] + 2 * ops.attribs['padding'][0][0] - var[str(ops.inputs['filter'])][3] + 1
            rank[2] = rank[2] + 2 * ops.attribs['padding'][0][0] - var[str(ops.inputs['filter'])][2] + 1
            rank[1] = copy.deepcopy(var[str(ops.inputs['filter'])][0])
            rank[0] = 1
            var[str(ops.outputs['output'])]=copy.deepcopy(rank)
            mat_flag=True
         elif ops.name=='relu':
            length = 1
            var[str(ops.outputs['y'])]=copy.deepcopy(var[str(ops.inputs['x'])])
            for i in range(len(var[str(ops.inputs['x'])])):
               length *= var[str(ops.inputs['x'])][i]
            if mat_flag and str(ops.inputs['x']) == mat_name:
               mat_name = str(ops.outputs['y'])
            else:
               self.source.write("static float "+str(ops.outputs['y'])+"["+str(length)+"]; \n")
         elif ops.name=='max_pool':
            length = 1
            rank=copy.deepcopy(var[str(ops.inputs['input'])])
            rank[3]=int(rank[3]/ops.attribs['size'][3])
            rank[2]=int(rank[2]/ops.attribs['size'][2])
            rank[1] = int(rank[1] / ops.attribs['size'][1])
            rank[0] = int(rank[0] / ops.attribs['size'][0])
            var[str(ops.outputs['output'])] = copy.deepcopy(rank)
            for i in range(len(var[str(ops.inputs['input'])])):
               length *= var[str(ops.inputs['input'])][i]
            if mat_flag and str(ops.inputs['input']) == mat_name:
               mat_name = str(ops.outputs['output'])
            else:
               self.source.write("static float "+str(ops.outputs['output'])+"["+str(length)+"]; \n")
         elif ops.name=='reshape':
            length = 1
            rank = copy.deepcopy(ops.attribs['shape'])
            for i in range(len(rank)):
               if rank[i]==-1:
                  rank[i]=1
            for i in range(len(rank)):
               length *= rank[i]
            var[str(ops.outputs['output'])] = copy.deepcopy(rank)
            if mat_flag and str(ops.inputs['input']) == mat_name:
               self.source.write("static float " + mat_name + "[" + str(length) + "]; \n")
               mat_flag=False
            self.source.write("static float " + str(ops.outputs['output']) + "[" + str(length) + "]; \n")
         elif ops.name=='matmul':
            if mat_flag:
               length = 1
               for i in range(len(var[mat_name])):
                  length *= var[mat_name][i]
               self.source.write("static float " + mat_name + "[" + str(length) + "]; \n")
               mat_flag=False
            rank=copy.deepcopy(var[ops.inputs['B']])
            rank[1] = copy.deepcopy(rank[0])
            rank[0] = copy.deepcopy(var[ops.inputs['A']][0])
            var[ops.outputs['C']]=copy.deepcopy(rank)
            self.source.write("static float " + str(ops.outputs['C']) + "[" + str(int(rank[1]*rank[0])) + "]; \n")
         elif ops.name=='add':
            assert var[ops.inputs['x']]==var[ops.inputs['y']]
            var[ops.outputs['z']]=copy.deepcopy(var[ops.inputs['x']])
            length = 1
            for i in range(len(var[ops.inputs['x']])):
               length *= var[ops.inputs['x']][i]
            if mat_flag and str(ops.inputs['x']) == mat_name and str(ops.outputs['z'])!='output':
               mat_name=str(ops.outputs['z'])
            self.source.write("static float "+str(ops.outputs['z'])+"["+str(length)+"]; \n")


      self.source.write("int main(){ \n\tint correct = 0; \n\tfor(int img = 0; img < " + str(
         int(self.test_end_id - self.test_start_id)) +
                        "; img++) {\n\t\tfloat *curr_im = input + img*")

      self.source.write(str(self.in_shape[2])+"*"+str(self.in_shape[3])+"*"+str(self.in_shape[1])+";\n")


      convolution={}
      for ops in self.graph.operations:
         if ops.name=='batch_normalization':
            if mat_flag:
               convolution['batchn']=[str(ops.inputs['mean']),str(ops.inputs['variance']),
                                      str(ops.inputs['offset']),str(ops.inputs['scale'])]
               mat_name=ops.outputs['output']
         if ops.name=='unsqueeze' or ops.name=='squeeze':
            self.source.write("\t\t"+str(ops.outputs['output'])+"="+str(ops.inputs['input'])+"; \n")
         elif ops.name=='mul':
            length=1
            for i in range(len(var[ops.inputs['y']])):
               length*=var[str(ops.inputs['y'])][i]
            self.source.write("\t\t fmul("+str(ops.inputs['y'])+", "+str(length)+","+str(ops.inputs['x'])+","+str(ops.outputs['z'])+"); \n")
         elif ops.name=='conv':
            if mat_flag:
               if convolution['act']=='input':
                  self.source.write("\t\t CnBnBwn(curr_im,"+convolution['ker']+","+
                                    str(var[convolution['act']][1])+","+str(var[convolution['act']][2])+
                                    ","+str(var[convolution['act']][3])+","+str(var[convolution['ker']][1])+
                                    ","+str(var[convolution['ker']][2])+","+str(var[convolution['ker']][3])+
                                    ","+str(var[convolution['ker']][0])+","+convolution['pad']+","+convolution['pooling']+
                                    ","+mat_name+", "+ convolution['bias']+","+ convolution['batchn'][0]+", "
                                    +convolution['batchn'][1]+","+ convolution['batchn'][3]+","+convolution['batchn'][2]
                                    +"); \n")
               else:
                  self.source.write("\t\t CnBnBwn(" + convolution['act'] + "," + convolution['ker'] + "," +
                                    str(var[convolution['act']][1]) + "," + str(var[convolution['act']][2]) +
                                    "," + str(var[convolution['act']][3]) + "," + str(var[convolution['ker']][1]) +
                                    "," + str(var[convolution['ker']][2]) + "," + str(var[convolution['ker']][3]) +
                                    "," + str(var[convolution['ker']][0]) + "," + convolution['pad'] + "," +
                                    convolution['pooling'] + "," + mat_name + ", "+ convolution['bias']+","+convolution['batchn'][0]+", "
                                    +convolution['batchn'][1]+","+ convolution['batchn'][3]+","+convolution['batchn'][2]
                                    +"); \n")
               if convolution['relu']:
                  self.source.write("\t\t frelu("+str(var[mat_name][1])+","+str(var[mat_name][2])+","+str(var[mat_name][3])+","+
                                    mat_name+","+mat_name+"); \n")
            convolution['act']=str(ops.inputs['input'])
            convolution['ker']=str(ops.inputs['filter'])
            convolution['pad']=str(ops.attribs['padding'][0][0])
            convolution['pooling']=str(1)
            convolution['relu']=False
            convolution['batchn']=["NULL","NULL","NULL","NULL"]
            convolution['bias']=ops.inputs['bias']
            mat_name=ops.outputs['output']
            mat_flag=True
         elif ops.name=='relu':
            if mat_flag and str(ops.inputs['x']) == mat_name:
               mat_name = str(ops.outputs['y'])
               convolution['relu']=True
            else:
               if len(var[ops.outputs['y']])==2:
                  self.source.write("\t\t frelu("+str(var[str(ops.outputs['y'])][0])+","
                                    +str(var[str(ops.outputs['y'])][1])+",0,"+str(ops.inputs['x'])+","+
                                    str(ops.outputs['y'])+"); \n")
               else:
                  self.source.write("\t\t frelu(" + str(var[str(ops.outputs['y'])][1]) + ","
                                    + str(var[str(ops.outputs['y'])][2]) + ","+str(var[str(ops.outputs['y'])][3])+","
                                    +str(ops.inputs['x']) +","+ str(ops.outputs['y']) + "); \n")
         elif ops.name=='max_pool':
            convolution['pooling']=str(ops.attribs['size'][3])
            if mat_flag and str(ops.inputs['input']) == mat_name:
               mat_name = str(ops.outputs['output'])
         elif ops.name=='reshape':
            if mat_flag:
               if convolution['act']=='input':
                  self.source.write("\t\t CnBnBwn(curr_im," + convolution['ker'] + "," +
                                    str(var[convolution['act']][1]) + "," + str(var[convolution['act']][2]) +
                                    "," + str(var[convolution['act']][3]) + "," + str(var[convolution['ker']][1]) +
                                    "," + str(var[convolution['ker']][2]) + "," + str(var[convolution['ker']][3]) +
                                    "," + str(var[convolution['ker']][0]) + "," + convolution['pad'] + "," +
                                    convolution['pooling'] +"," + mat_name + ", " + convolution['bias'] + ","
                                    + convolution['batchn'][0] + ", "+ convolution['batchn'][1] + ","
                                    + convolution['batchn'][3] + "," +convolution['batchn'][2]+ "); \n")
               else:
                  self.source.write("\t\t CnBnBwn(" + convolution['act'] + "," + convolution['ker'] + "," +
                                    str(var[convolution['act']][1]) + "," + str(var[convolution['act']][2]) +
                                    "," + str(var[convolution['act']][3]) + "," + str(var[convolution['ker']][1]) +
                                    "," + str(var[convolution['ker']][2]) + "," + str(var[convolution['ker']][3]) +
                                    "," + str(var[convolution['ker']][0]) + "," + convolution['pad'] + "," +
                                    convolution['pooling'] + "," + mat_name + ", " + convolution['bias'] + "," +
                                    convolution['batchn'][0] + ", "+ convolution['batchn'][1] + ","
                                    + convolution['batchn'][3] + "," +convolution['batchn'][2]+ "); \n")
               if convolution['relu']:
                  self.source.write("\t\t frelu("+str(var[mat_name][1])+","+str(var[mat_name][2])+","+str(var[mat_name][3])+","+
                                    mat_name+","+mat_name+"); \n")
               self.source.write("\t\t freshape("+mat_name+","+str(var[mat_name][3])+","+str(var[mat_name][2])+","+str(var[mat_name][1])+","+
                                    ops.outputs['output']+"); \n")
               mat_flag=False
         elif ops.name=='matmul':
            assert var[ops.inputs['A']][1]==var[ops.inputs['B']][1]
            self.source.write("\t\t FcBnXnorArrNoBin(" + ops.inputs['A'] + "," + ops.inputs['B'] + "," + str(
                  var[ops.inputs['B']][1]) +
                                 "," + str(var[ops.inputs['B']][0]) + "," + str(
                  ops.outputs['C']) + "," + "NULL, NULL, NULL, NULL); \n")
         elif ops.name=='add':
            assert var[ops.inputs['x']]==var[ops.inputs['y']]
            self.source.write("\t\t fadd("+str(ops.inputs['x'])+","+str(ops.inputs['y'])+","+str(var[ops.outputs['z']][0])
                              +","+str(var[ops.outputs['z']][1])+",0,"+str(ops.outputs['z'])+"); \n")


      #testing and inference
      self.source.write("\t\t float max = -INFINITY; \n\t\tint maxIdx = 0; \n\t\tfor (int i = 0; i <10; i++) { \n\t\t\t printf(\"%f, \", output[i]);\n\t\t\t if (output[i] > max) { \n\t\t\t\t max = output[i]; \n\t\t\t\t")
      self.source.write("maxIdx = i;\n\t\t\t }\n\t\t}\n\t\t")
      self.source.write("printf(\"\\n\");")
      self.source.write("printf(\"Image %d: label: %d, actual: %d\\n\",img, maxIdx, labels[img]); \n\t\t")
      self.source.write("if (maxIdx == labels[img]) correct += 1; \n\t}\n\tprintf(\"Accuracy: %f%%\\n\", 100.0*(float)correct/"
                        +str(int(self.test_end_id-self.test_start_id))+"); \n\treturn (EXIT_SUCCESS); \n}")
      self.source.close()


   def write_images(self):
      '''
         write out images for both testing and inference

      :return: NA
      '''
      image=open('image.h','w+')
      os.chdir('..')

      transform_train = transforms.Compose([
         transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])

      transform_test = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])

      trainset = torchvision.datasets.CIFAR10(
         root='./data', train=True, download=True, transform=transform_train)

      testset = torchvision.datasets.CIFAR10(
         root='./data', train=False, download=True, transform=transform_test)
      testloader = torch.utils.data.DataLoader(
         testset, batch_size=1, shuffle=False, num_workers=2)
      '''
      transform = transforms.Compose(
         [transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
      testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform)
      testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                               shuffle=False, num_workers=2)
      '''
      label=testloader.dataset.targets
      image.write('#define IMAGES {\\\n')
      count=0
      '''with torch.no_grad():
         for data in testloader:
            images, labels = data'''
      with torch.no_grad():
         for batch_idx, (inputs, targets) in enumerate(testloader):
            rank=inputs.shape
            for y in range(rank[2]):
               for x in range(rank[3]):
                  for z in range(rank[1]):
                     image.write(str(format(inputs[0][z][y][x].item(),'.4f')) + ', ')
            image.write('\\\n')
            count+=1
            if count==100:
               break;
         image.write('}\n')
         image.write('#define LABELS {\\\n')
      for i in range(self.test_end_id-self.test_start_id):
         image.write(str(label[i])+', ')
      image.write('}\n')
      image.close()

   def write_last_layer(self):
      if len(self.tempweight) != 0:
         tt = np.copy(self.tempweight)
         if len(tt.shape) == 4:
            self.writecn(True, [tt.shape[3], tt.shape[2], tt.shape[1], tt.shape[0]],
                              self.tempweight, self.tempsparse, self.tempoutput, self.name)
         else:
            self.writefc(True, tt.shape, self.tempweight, self.tempsparse, self.tempoutput,
                              self.name)

def main():
   print("WARNING: Current 3PXNet inference library does not support operations "
         "other than convolution or matrix multiplication")
   print("All other operations will be skipped.")
   # Argument parsing
   parser = argparse.ArgumentParser(description='automatically generate inference code')
   #  WJR: what should be in the directory? What is the expected input format?"
   # FZ: err I think this help is clear enough. As long as the user makes no modifications to the converted NNEF
   # formatted directory, it is fine.
   parser.add_argument('--input', help="""name of input directory. This should be the converted NNEF "
      "formatted neural network which ends with .nnef with no other modifications. Example: --input=FC_Small.nnef""")
   parser.add_argument('--dataset', metavar='DATASET', default='MNIST',
                       help='Dataset to train on. Currently choose from MNIST and CIFAR10')
   parser.add_argument('--test_start_id',default=0,help='the starting index of dataset for testing')
   parser.add_argument('--test_end_id', default=100, help='the ending index of dataset for testing')
   args = parser.parse_args()
   dataset = args.dataset
   input_dir = args.input
   test_start_id=args.test_start_id
   test_end_id=args.test_end_id

   converter=convert(input_dir,dataset,test_start_id,test_end_id)
   # WJR: What are all those functions doing?
   # FZ: Yeah I write them in their declaration's comments
   # WJR: Make a comment before each one of them here. This will explain the whole flow of the converstion
   # process without having to look at each instructions. Comment redundancy is good, to a degree.
   # FZ: fixed
   #load the nnef graph into compiler
   converter.loadgraph()
   # the last layer will not be binarized, so its batch norm has to be dealt differently
   # this function finds such batch norm operation
   converter.find_batch_last()
   # write included headers into source code
   converter.write_source_first()
   # for each operation shown in the graph, compile it
   converter.processing_graph()
   # WJR: what is happening here? Why does the last layer need all this extra code?
   # FZ: This is all because of permutation.We need the every next layer's permute list to permute the current layer
   # FZ: along output channel. So I store every layer as a temp layer in the object. This causes the last layer to be
   # FZ: stored in the object but not written out.
   # WJR: Then write the class function to deal with it. Don't clutter main with extra code.
   # FZ: fixed
   # the last layer should be written out as well
   converter.write_last_layer()
   # calculate batch normalization threshold and sign
   converter.calculate_batch()
   # write out all remaining source code
   converter.write_source_second()
   # write out images for both inference and testing
   converter.write_images()

if __name__ == '__main__':
   main()