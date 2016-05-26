import pandas as pd
import numpy as np

########################################################################################################################
# Load the data as a Pandas DataFrame

persons = pd.read_csv('/Users/johnkimnguyen/Desktop/Bud Lab/temp/person.txt', sep='\t', header= None, names = ['person_id', 'name'])
degrees  = pd.read_csv('/Users/johnkimnguyen/Desktop/Bud Lab/temp/degree.txt', sep='\t', header= None, names = ['degree_id', 'title'])
relations = pd.read_csv('/Users/johnkimnguyen/Desktop/Bud Lab/temp/relation.txt', sep='\t', header= None, names = ['degree_id', 'relation_id', 'score'])
terms = pd.read_csv('/Users/johnkimnguyen/Desktop/Bud Lab/temp/term.txt', sep='\t', header= None, names = ['term_id', 'term'])
schools = pd.read_csv('/Users/johnkimnguyen/Desktop/Bud Lab/temp/school.txt', sep='\t', header= None, names = ['school_id', 'school'])

########################################################################################################################
# Create the commuting matrix M(i,:) and perform reordering
# Parameters: adjacency matrices for the meta-path and person ID
def commute_matrix(adj1, adj2, person_id):
    M_p = adj1.dot(adj2)

    order = M_p.index.tolist()
    order.insert(0, order.pop(order.index(person_id)))
    M_pi = M_p.reindex(order)

    M = M_pi.dot(M_p.transpose())
    mid = M[person_id]
    M.drop(labels=[person_id], axis=1,inplace = True)
    M.insert(0, person_id, mid)
    return M

# Get the scaling terms to compute the PathSim scores
# Parameters: commuting matrix and person id
def scaling(M, person_id):
    # Diagonal vector
    D = list(np.diagonal(M))
    scale = []
    for i in D:
        scale.append(2 / (D[0] + i))
    CF = pd.DataFrame(M.ix[person_id])
    CF.columns = ['pathsim_score']
    s = pd.Series(scale)
    return CF, s

# Utility function to perform element-wise multiplication between Dataframes and Series
mult = lambda x: np.asarray(x) * np.asarray(s)

########################################################################################################################
#  Create the person-degree adjacency

merged = relations.merge(persons, left_on='relation_id', right_on='person_id')
person_degree = merged[['degree_id', 'person_id']]
person_degree_adj = pd.DataFrame(0, index=persons['person_id'], columns=degrees['degree_id'])
for degree, person in zip(person_degree['degree_id'], person_degree['person_id']):
    person_degree_adj.set_value(person, degree, 1)
person_degree_adj = person_degree_adj.dropna(axis='columns')


# Create the degree-Venue adjacency
merged = relations.merge(schools, left_on='relation_id', right_on='school_id')
degree_venue = merged[['degree_id', 'school_id']]
degree_venue_adj = pd.DataFrame(0, index=degrees['degree_id'], columns=schools['school_id'])
for degree, venue in zip(degree_venue['degree_id'], degree_venue['school_id']):
    degree_venue_adj.set_value(degree, venue, 1)
degree_venue_adj = degree_venue_adj.dropna(axis='index')

#  Create the degree-Term adjacency
merged = relations.merge(terms, left_on='relation_id', right_on='term_id')
degree_term = merged[['degree_id', 'term_id']]
degree_term_adj = pd.DataFrame(0, index=degrees['degree_id'], columns=terms['term_id'])
for degree, term in zip(degree_term['degree_id'], degree_term['term_id']):
    degree_term_adj.set_value(degree, term, 1)
degree_term_adj = degree_term_adj.dropna(axis='index')

########################################################################################################################
# Kristin Faison id: 55154 | use meta-path Person-Degree-School-Degree-School
# Master - Forest and Wood Sciences | PhD - English and Speech Teacher Education

# Create output files
anhai_pathsim = open('anhai_pathsim.txt', 'w')

########################################################################################################################
# Get the scores

M_anhai = commute_matrix(person_degree_adj, degree_venue_adj, 55154)
AD, s = scaling(M_anhai, 55154)
an_hai = AD.apply(mult)
an_hai_result = persons.merge(an_hai, left_on='person_id', right_index =True)
an_hai_result = an_hai_result.sort_values('pathsim_score', ascending=False)
print(an_hai_result.head(12), file = anhai_pathsim)
print(an_hai_result)