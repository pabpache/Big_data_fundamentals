#Name: Pablo Pacheco
#znumber: z5222810

from pyspark import SparkConf, SparkContext

########## Question 1 ##########
# do not change the heading of the function

def c2lsh(data_hashes, query_hashes, alpha_m, beta_n):
 
    #make an RDD with every row as: (id, list, minimum offset to be a candidate)
    rdd_offset = data_hashes.map(lambda x: (x[0],x[1],min_offset(x[1],query_hashes,alpha_m)))
    maximum = rdd_offset.map(lambda x:x[2]).max()     #maximum offset (necessary to be a candidate) for any row in data_hashes RDD
    minimum = rdd_offset.map(lambda x:x[2]).min()     #The minimum offset necessary to get at least one candidate from data_hashes
    
    #Find the offset that gives the enough quantity of candidates (USING BINARY SEARCH)
    while minimum<=maximum:
        mid=(minimum+maximum)//2            #'mid' is going to be the offset to check in this iteration
        rdd_curr= data_hashes.map(lambda x: (x[0],count_tag(x[1],query_hashes,mid,alpha_m))).filter(lambda x: x[1]==1)
        # a row in rdd_curr is (int,tag(1 o 0))
        curr_count=rdd_curr.count()     #number of candidates considering the current offset
        if mid==0:           #this is when the offset is zero
            if curr_count>=beta_n:
                break        #using an offset of 0 it is possible get the required number of candidates
            else:
                minimum= mid+1
        elif curr_count>=beta_n:
            #check if an offset of 'mid-1' meet the necessary number of candidates as well
            if data_hashes.map(lambda x: (x[0],count_tag(x[1],query_hashes,mid-1,alpha_m))).filter(lambda x: x[1]==1).count() < beta_n:
                break      #the optimal offset was found
            else:              #there is a lower offset that meets the requiered number of candidates
                maximum = mid-1
        else:
            minimum = mid+1

       
    return rdd_curr.map(lambda x: x[0])


#The count_tag function returns 1 when data_vector is a candidate of query_vector given an offset and an alpha_m
#otherwise returns 0
def count_tag(data_vector,query_vector,offset,alpha_m):
    counter=0
    for i in range(len(query_vector)):    #length of query_vector and data_vector is the same
        if abs(data_vector[i]-query_vector[i]) <= offset:
            counter +=1
    if counter<alpha_m:
        return 0
    else:
        return 1
   

#The min_offset function returns the offset necessary to make the vector a candidate of the query given an alpha_m
def min_offset(data_vector,query_vector,alpha_m):
    #calculate the max difference between the components of data_vector and query_vector
    length= len(query_vector)
    max_difference=0
    for i in range(length):
        if abs(data_vector[i]-query_vector[i])>max_difference:
            max_difference= abs(data_vector[i]-query_vector[i])
    
    #using binary search find the offset necessary to make data_vector a candidate of query_vector
    maximum=max_difference
    minimum=0
    mid=0
    while minimum<=maximum:
        mid=(maximum + minimum)//2
        curr_tag = count_tag(data_vector,query_vector,mid,alpha_m)    #this is '1' if the vector is a candidate with mid as the offset
        if mid==0:
            if curr_tag==1:
                return mid
            else:
                minimum=mid+1          
        elif curr_tag==1:
            if count_tag(data_vector,query_vector,mid-1,alpha_m)==0:
                return mid       #the minimal offset to be a candidate was reached
            else:                #There is an offset less than mid that also make the vecor a candidate
                maximum = mid-1
        else:       #a higher offset is necessary to make the vector a candidate
            minimum = mid+1
        
    #The 'while' loop should always give a result, that's why there is no a return at the end    
        

    

        
  
