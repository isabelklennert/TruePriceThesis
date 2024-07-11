#Import packages
from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
import networkx as nx
import numpy as np
from mesa.datacollection import DataCollector
from numpy.random import choice
from sklearn.neighbors import KernelDensity
import os
import joblib
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import bisect
import scipy.spatial
import random

class ConsumerAgent(Agent):
    def __init__(self, unique_id, model, budget, preference_sustainability, preference_conformity, stability_job, initial_product, weight_financial, weight_social, weight_personal):
        super().__init__(unique_id, model)
        self.budget = budget
        self.preference_sustainability = preference_sustainability
        self.preference_conformity = preference_conformity
        self.stability_job = stability_job
        self.last_purchased_product = initial_product
        self.chosen_product = initial_product
        self.is_currently_using_true_price = False #Initially no one is using TP
        self.previous_true_price_usage = False  #No one has used TP in the past
        self.has_used_true_price = False    
        self.decision_mode = None
        self.weight_financial = weight_financial
        self.weight_social = weight_social
        self.weight_personal = weight_personal
        self.neighbors = []
       
        self.choice_changes = 0
        self.decision_mode_changes = 0
        self.previous_decision_mode = None   
 
        # Initialize satisfaction and uncertainty for each need
        self.F_satisfaction = np.random.uniform(0.01, 1)  
        self.F_uncertainty = np.random.uniform(0.01, 1)
        self.S_satisfaction = np.random.uniform(0.01, 1)
        self.S_uncertainty = np.random.uniform(0.01, 1)
        self.P_satisfaction = np.random.uniform(0.01, 1)  
        self.P_uncertainty = np.random.uniform(0.01, 1)
 
    def step(self):

        self.update_true_price_usage_status() #Whether TP was ever used
        self.previous_true_price_usage = self.is_currently_using_true_price #Whether a TP product was being used in the previous iteration before any potential changes
        self.calculate_satisfaction_uncertainty(self.chosen_product) #Calculate satisfaction and uncertainty for the chosen product from the previous round
 
        satisfied = self.is_satisfied() >= self.model.satisfaction_threshold
        #Uncertainty: 0 when certain, 1 when uncertain
        uncertain = self.is_uncertain() >= self.model.uncertainty_threshold
 
        previous_product = self.chosen_product
        self.previous_decision_mode = self.decision_mode

        if satisfied and not uncertain:
            self.repeat_action()
        elif satisfied and uncertain:
            self.imitation()
        elif not satisfied and not uncertain:
            self.deliberation()  
        elif not satisfied and uncertain:
            self.social_comparison()

        if self.chosen_product != previous_product:
            self.choice_changes += 1

        if self.previous_decision_mode != self.decision_mode:
            self.decision_mode_changes += 1
       
        self.last_purchased_product = self.chosen_product
 
       
       
    def update_true_price_usage_status(self):
        # Update the True Price usage status based on chosen_product
        if self.chosen_product:
            self.is_currently_using_true_price = self.chosen_product.is_true_price  #Sets value of chosen product to true/false
            if self.is_currently_using_true_price:  #checks if current product is TP
                self.has_used_true_price = True     #if yes then has used true price is also true
        else:
            self.is_currently_using_true_price = False      #if no then hasn't used true price
 
    def calculate_satisfaction_uncertainty(self, product):
        # Calculate financial, social, and personal satisfaction and uncertainty
        self.calculate_financial_satisfaction_and_uncertainty(product)
        self.calculate_social_satisfaction_and_uncertainty(product)
        self.calculate_personal_satisfaction_and_uncertainty(product)
 
         
    def calculate_financial_satisfaction_and_uncertainty(self, product):
        # Calculate the cost of the product
        product_cost = product.normal_price

        # Check for negative product cost
        if product_cost < 0:
            raise ValueError(f"Product cost is negative: {product_cost} for product ID: {product.product_id}")

        if self.budget < 0:
            raise ValueError(f"Budget is negative: {self.budget} for agent ID: {self.unique_id}")

        # Calculate proportion spent
        proportion_spent = product_cost / self.budget if self.budget > 0 else 1

        # Financial Satisfaction: Non-linear function, decreases more sharply as cost approaches and exceeds budget
        # Satisfaction can range between 0 and 1
        self.F_satisfaction = max(0, min(1, 1 - (proportion_spent ** 2) / (1 + proportion_spent ** 2))) if proportion_spent <= 1 else max(0, min(1, 1 / proportion_spent ** 2))

        # Financial Uncertainty: Influenced by inflation rate, proportion of budget spent, and stability of job
        job_stability_factor = 1 - self.stability_job  # 1 is fully stable so then less uncertain
        inflation_impact = self.model.inflation_rate / 100

        # Adjust the impact of proportion spent and inflation based on job stability
        proportion_impact = proportion_spent * (1 + job_stability_factor)
        inflation_impact_adjusted = inflation_impact * (1 + job_stability_factor)

        # Combine factors to calculate financial uncertainty
        combined_effect = proportion_impact + inflation_impact_adjusted
        self.F_uncertainty = max(0, min(1, combined_effect / 2))

        # Print diagnostic information
        #print(f"Agent {self.unique_id}:")
        #print(f"  Budget: {self.budget}")
        #print(f"  Product Cost: {product_cost}")
        #print(f"  Proportion Spent: {proportion_spent}")
        #print(f"  Financial Satisfaction: {self.F_satisfaction}")
        #print(f"  Financial Uncertainty: {self.F_uncertainty}")
 

    def calculate_social_satisfaction_and_uncertainty(self, product):

        # Get neighbors from the social network
        neighbors = self.neighbors
 
        # Calculate conformity satisfaction
        same_product_count = sum(1 for neighbor in neighbors if neighbor.previous_true_price_usage == self.previous_true_price_usage) #how many neighbours have the same TP usage as the current agent
        conformity = same_product_count / len(neighbors) if neighbors else 0 #proportion of such neighbours among all neighbours
        # Diminish the impact of neighbors conforming to the same product
        self.S_satisfaction = max(0, min(1, (conformity * self.preference_conformity) ** 0.5)) #square root: as conformity increases, each additional 'unit' of conformity contributes less to overall satisfaction
 
        # Calculate social uncertainty
        previous_true_price_users = {neighbor for neighbor in neighbors if neighbor.previous_true_price_usage} #used TP in previous time step
        past_true_price_users = {neighbor for neighbor in neighbors if neighbor.has_used_true_price} #have ever used TP
   
        if len(past_true_price_users) > 0:
            # Variability of neighbors' choices affects uncertainty
            self.S_uncertainty = 1 - len(previous_true_price_users) / len(past_true_price_users)    #proportion of neighbours who have changed their behaviour
        else:
            self.S_uncertainty = self.preference_conformity #if no one has used TP then you are uncertain if you have a high preference for conformity
        self.S_uncertainty = max(0, min(1, self.S_uncertainty))
 
   
    def calculate_personal_satisfaction_and_uncertainty(self, product):

        # Personal Satisfaction: Based on the 'green value' of the product
        # Assuming each product has a 'green_score' attribute (0 to 1 scale)
        self.P_satisfaction = max(0, min(1, self.preference_sustainability * product.green_score))
 

        remediation_uncertainty = product.remediation_level * self.preference_sustainability
        self.P_uncertainty = 1-remediation_uncertainty
        self.P_uncertainty = max(0, min(1, self.P_uncertainty))
 

    def calculate_weighted_average(self, values, weights=None):
        if weights is None:
            # If no weights provided, use equal weighting
            weights = [1 for _ in values]
        weighted_sum = sum(w * v for w, v in zip(weights, values))
        total_weight = sum(weights)
        return weighted_sum / total_weight if total_weight != 0 else 0
 
    def is_satisfied(self):
        total_satisfaction = self.calculate_weighted_average(
            [self.F_satisfaction, self.S_satisfaction, self.P_satisfaction],
            [self.weight_financial, self.weight_social, self.weight_personal]
        )
        return max(0, min(1, total_satisfaction))
 

    def is_uncertain(self):
        total_uncertainty = self.calculate_weighted_average(
            [self.F_uncertainty, self.S_uncertainty, self.P_uncertainty],
            [self.weight_financial, self.weight_social, self.weight_personal]
        )
        return max(0, min(1, total_uncertainty))
 
    def repeat_action(self):    #individual and automated
        self.decision_mode = "repeat"
        # Stick with the same product as last time
        if self.last_purchased_product is not None:
            self.chosen_product = self.last_purchased_product
        else:
            # Fallback logic if no product was chosen before
            self.chosen_product = np.random.choice(self.model.products)
   
    def deliberation(self): #individual and reasoned
        self.decision_mode = "deliberate"
        # Evaluate all products and choose the one with the highest weighted score (financial and personal) - not social because deliberation is an individual process)
        self.chosen_product = self.choose_product()
 
    def imitation(self):    #social and automated
        self.decision_mode = "imitate"
        neighbor_product_count = {}
        tp_product_count = 0
        non_tp_product_count = 0
        for neighbor in self.neighbors:
            product = neighbor.last_purchased_product
            if product:
                neighbor_product_count[product] = neighbor_product_count.get(product, 0) + 1
                if product.is_true_price:
                    tp_product_count += 1
                else:
                    non_tp_product_count += 1
 
        if neighbor_product_count:
            max_count = max(neighbor_product_count.values())
            most_popular_products = [product for product, count in neighbor_product_count.items() if count == max_count]
 
            if len(most_popular_products) > 1:
                # Tie between products, select randomly from tied products
                self.chosen_product = np.random.choice(most_popular_products)
            else:
                # Clear popular product
                self.chosen_product = most_popular_products[0]
        else:
            # No clear popular choice, pick randomly from the category with more products
            if tp_product_count > non_tp_product_count:
                tp_products = [product for product in self.model.products if product.is_true_price]
                self.chosen_product = np.random.choice(tp_products)
            elif non_tp_product_count > tp_product_count:
                non_tp_products = [product for product in self.model.products if not product.is_true_price]
                self.chosen_product = np.random.choice(non_tp_products)
            else:
                # Equal number of TP and non-TP products, select randomly from all
                self.chosen_product = np.random.choice(self.model.products)
 

    def social_comparison(self):    #social and reasoned
        self.decision_mode = "social_compare"
        comparable_neighbors = [        #identifies similar neighbours
            neighbor for neighbor in self.neighbors
            if self.is_comparable(neighbor)
        ]
 
        product_count = {}
        for neighbor in comparable_neighbors:
            product = neighbor.last_purchased_product
            if product:
                product_count[product] = product_count.get(product, 0) + 1
       
        if product_count:
            max_count = max(product_count.values())
            most_popular_products = [product for product, count in product_count.items() if count == max_count]
 
            if len(most_popular_products) == 1:
                # Clear popular product
                self.chosen_product = most_popular_products[0]
            else:
                # Deliberate among popular or tied products
                self.deliberate_among_choices(most_popular_products)
        else:
            # No clear popular choice, deliberate among all available products
            self.chosen_product = self.deliberate_among_choices(self.model.products)



    def is_comparable(self, other_agent):
        for attribute in self.model.comparison_attributes:
            self_attr_value = getattr(self, attribute)
            other_attr_value = getattr(other_agent, attribute)
            if abs(self_attr_value - other_attr_value) > self.model.epsilon * self_attr_value:
                return False
        return True
 
    def deliberate_among_choices(self, products):
        best_product = None
        best_score = -float('inf')
        for product in products:
            self.calculate_financial_satisfaction_and_uncertainty(product)
            self.calculate_personal_satisfaction_and_uncertainty(product)
            self.calculate_social_satisfaction_and_uncertainty(product)
 
            # Combine the satisfaction and uncertainty scores
            net_financial_score = self.F_satisfaction - self.F_uncertainty
            net_personal_score = self.P_satisfaction - self.P_uncertainty
            net_social_score = self.S_satisfaction - self.S_uncertainty
 
            # Calculate the weighted average
            combined_score = self.calculate_weighted_average(
                [net_financial_score, net_personal_score, net_social_score],
                [self.weight_financial, self.weight_personal, self.weight_social]
            )
 
            if combined_score > best_score:
                best_score = combined_score
                best_product = product
        if best_product is None and products:
            best_product = np.random.choice(products)

        return best_product
 
    def get_neighbors(self):
        neighbors = []
        neighbor_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        
        for node in neighbor_nodes:
            cell_contents = self.model.grid.get_cell_list_contents([node])
            if cell_contents:  # Check if the list is not empty
                neighbors.append(cell_contents[0])  # Assuming there's at most one agent per node
        return neighbors
 
    def choose_product(self):
        # Filter products based on availability
        available_products = self.model.products
        #print(f"Available products: {len(available_products)}")

        #if not available_products:
            #print("No available products, assigning default.")  # Debugging print

        # Choose a product from available products based on a composite score of financial and personal satisfaction
        best_product = None
        best_score = -1
        for product in available_products:
            # Calculate satisfaction and uncertainty for each product
            self.calculate_financial_satisfaction_and_uncertainty(product)
            self.calculate_personal_satisfaction_and_uncertainty(product)
 
            # Combine the satisfaction and uncertainty scores
            net_financial_score = self.F_satisfaction - self.F_uncertainty
            net_personal_score = self.P_satisfaction - self.P_uncertainty
 
            total_weight = self.weight_financial + self.weight_personal
            normalized_weight_financial = self.weight_financial / total_weight
            normalized_weight_personal = self.weight_personal / total_weight
 
            combined_score = self.calculate_weighted_average(
                [net_financial_score, net_personal_score],
                [normalized_weight_financial, normalized_weight_personal]
            )
            if combined_score > best_score:
                best_score = combined_score
                best_product = product
        if best_product is None:
            best_product = np.random.choice(available_products)
        return best_product
 

    @property
    def is_using_true_price(self):
        return self.is_currently_using_true_price




class Product:
    def __init__(self, product_id, is_true_price, normal_price, price_increase_percentage, green_score, remediation_level):
        self.product_id = product_id
        self.is_true_price = is_true_price
        self.normal_price = normal_price
        self.price_increase_percentage = price_increase_percentage
        self.green_score = green_score
        self.remediation_level = remediation_level
        self.tp_available = False  # Initially, TP is not available
        
       



class ConsumatModel(Model):
    def __init__(self, config):
        np.random.seed(config['seed'])
        random.seed(config['seed'])
        self.config = config
        self.num_agents = config['network_params']['n']
        self.TP_percentage = config['TP_percentage']
        self.product_price_range = config['product_price_range']
        self.num_products = config['num_products']
        self.schedule = RandomActivation(self)
        self.min_increase_percentage = config['min_increase_percentage']
        self.max_increase_percentage = config['max_increase_percentage']
        self.satisfaction_threshold = config['satisfaction_threshold']
        self.uncertainty_threshold = config['uncertainty_threshold']
        self.inflation_rate = config['inflation_rate']
        self.true_price_introduced = False
        self.homophily_attribute = config['homophily_attribute']
        self.rewiring_probability = config['network_params'].get('rewiring_probability', 0)  # Default to 0 if not present
        self.epsilon = config['epsilon']
        self.comparison_attributes = config['comparison_attributes']


        # Load the datasets
        self.income_distribution = pd.read_csv('/Applications/UNI/ThesisNew/datasets/Adjusted_Distribution_of_spendable_income_2022.csv')
        self.processed_df = pd.read_csv('/Applications/UNI/ThesisNew/datasets/processed_dutch_population.csv')


        if self.num_agents > len(self.processed_df):
            raise ValueError("Number of agents exceeds the number of rows in the dataset")

        # Adjust the income distribution to fit within the product price range
        self.adjust_income_distribution_to_price_range()

        # Create products
        self.products = self.create_products()
        self.agents = self.create_agents()

    
        self.G = self.create_network(config['network_type'], config['network_params'])
        self.grid = NetworkGrid(self.G)
        self.place_agents_on_grid()
        self.rewire_network_based_on_homophily()
        self.update_agents_neighbors()

        self.datacollector = DataCollector(
            model_reporters={
                #"Min_Satisfaction": lambda model: min(agent.is_satisfied() for agent in model.schedule.agents),
                #"Max_Satisfaction": lambda model: max(agent.is_satisfied() for agent in model.schedule.agents),
                #"Min_Uncertainty": lambda model: min(agent.is_uncertain() for agent in model.schedule.agents),
                #"Max_Uncertainty": lambda model: max(agent.is_uncertain() for agent in model.schedule.agents),
                "True_Price_Adoption_Rate": lambda model: model.calculate_true_price_adoption_rate(),
                "Average_Satisfaction": lambda model: np.mean([agent.is_satisfied() for agent in model.schedule.agents]),
                "Average_Choice_Changes": lambda model: np.mean([agent.choice_changes for agent in model.schedule.agents]),
                "Average_Decision_Mode_Changes": lambda model: np.mean([agent.decision_mode_changes for agent in model.schedule.agents]),
                "Average Degree": lambda model: np.mean([deg for n, deg in model.G.degree()]),
                "Variance of Degree": lambda model: np.var([deg for n, deg in model.G.degree()]),
                "Average Clustering": lambda model: nx.average_clustering(model.G),
                "Average Path Length": lambda model: nx.average_shortest_path_length(model.G) if nx.is_connected(model.G) else None,
                "Avg_F_Satisfaction": lambda model: np.mean([agent.F_satisfaction for agent in model.schedule.agents]),
                "Avg_S_Satisfaction": lambda model: np.mean([agent.S_satisfaction for agent in model.schedule.agents]),
                "Avg_P_Satisfaction": lambda model: np.mean([agent.P_satisfaction for agent in model.schedule.agents]),
                "Avg_F_Uncertainty": lambda model: np.mean([agent.F_uncertainty for agent in model.schedule.agents]),
                "Avg_S_Uncertainty": lambda model: np.mean([agent.S_uncertainty for agent in model.schedule.agents]),
                "Avg_P_Uncertainty": lambda model: np.mean([agent.P_uncertainty for agent in model.schedule.agents])
            },
            agent_reporters={
                "True_Price_Usage": "is_using_true_price",
                "Repeat": lambda a: a.decision_mode == "repeat",
                "Imitate": lambda a: a.decision_mode == "imitate",
                "Deliberate": lambda a: a.decision_mode == "deliberate",
                "Social_Compare": lambda a: a.decision_mode == "social_compare",
                "Satisfaction": lambda a: a.is_satisfied(),
                "Uncertainty": lambda a: a.is_uncertain(),
                "Choice_Changes": "choice_changes",
                "Decision_Mode_Changes": "decision_mode_changes"
            }
        )
    
    def create_network(self, network_type, params):
        np.random.seed(self.config['seed'])
        if network_type == 'watts_strogatz':
            return nx.watts_strogatz_graph(n=params['n'], k=params['k'], p=params['rewiring_probability'])
        elif network_type == 'barabasi_albert':
            return nx.barabasi_albert_graph(n=params['n'], m=params['m'])
        elif network_type == 'random_regular':
            return nx.random_regular_graph(d=params['d'], n=params['n'])
        elif network_type == 'holme_kim':
            return nx.powerlaw_cluster_graph(n=params['n'], m=params['m'], p=params['rewiring_probability'])
        else:
            raise ValueError(f"Unsupported network type: {network_type}")


    def rewire_network_based_on_homophily(self):
        np.random.seed(self.config['seed'])
        random.seed(self.config['seed'])
        alpha = self.config.get('alpha', 1.0)
        beta = self.config.get('beta', 0.5)

        G = self.G.copy()

        # Rewiring process based on homophily attribute
        for node in list(G.nodes):
            neighbors = list(G.neighbors(node))
            for neighbor in neighbors:
                attribute_difference = abs(getattr(self.agents[node], self.homophily_attribute) - getattr(self.agents[neighbor], self.homophily_attribute))
                P_d = 1 / (1 + (attribute_difference / beta) ** alpha)
                if np.random.rand() > P_d:
                    G.remove_edge(node, neighbor)
                    second_order_neighbors = set()
                    for first_order in G.neighbors(node):
                        second_order_neighbors.update(G.neighbors(first_order))
                    second_order_neighbors.discard(node)
                    second_order_neighbors.difference_update(G.neighbors(node))
                    if second_order_neighbors:
                        attribute_distances = [abs(getattr(self.agents[node], self.homophily_attribute) - getattr(self.agents[second_order], self.homophily_attribute)) for second_order in second_order_neighbors]
                        probabilities = [1 / (1 + (d / beta) ** alpha) for d in attribute_distances]
                        total_prob = sum(probabilities)
                        probabilities = [prob / total_prob for prob in probabilities]
                        new_neighbor = np.random.choice(list(second_order_neighbors), p=probabilities)
                        G.add_edge(node, new_neighbor)

        self.G = G



    def adjust_income_distribution_to_price_range(self):
        product_price_range = self.product_price_range
        min_price = product_price_range[0]
        max_price = product_price_range[1]

        # Calculate the maximum possible price increase
        max_possible_price = max_price * (1 + self.max_increase_percentage / 100)

        # Separate negative and non-negative incomes
        non_negative_income = self.income_distribution['Income in thousands'].clip(lower=0)
        negative_income_mask = self.income_distribution['Income in thousands'] < 0

        # Normalize the non-negative part of the income distribution to [0, 1]
        normalized_income = (non_negative_income - non_negative_income.min()) / (non_negative_income.max() - non_negative_income.min())
    
        # Scale to fit the extended product price range
        self.income_distribution['Transformed Income'] = min_price + normalized_income * (max_possible_price - min_price)
    
        # Handle negative incomes by assigning a value less than min_price
        self.income_distribution.loc[negative_income_mask, 'Transformed Income'] = min_price - 1

    def update_agents_neighbors(self):
            for agent in self.agents:
                agent.neighbors = self.get_neighbors_for_agent(agent)
        
            
    def get_neighbors_for_agent(self, agent):
        neighbors = []
        neighbor_nodes = self.grid.get_neighbors(agent.pos, include_center=False)
        
        for node in neighbor_nodes:
            cell_contents = self.grid.get_cell_list_contents([node])
            if cell_contents:  # Assuming there's at most one agent per node
                neighbors.extend(cell_contents)  # You might need to filter for agents only, depending on your grid's structure

        return neighbors
    


 
    def create_agents(self):
        np.random.seed(self.config['seed'])
        random.seed(self.config['seed'])
        temp_agents = []
        sampled_budgets = []


        for i in range(self.num_agents):
            # Sampling budget from the income distribution
            budget = np.random.choice(
                self.income_distribution['Transformed Income'].values,
                p=self.income_distribution['Probability'].values
            )
            sampled_budgets.append(budget)  # Collect sampled budget for diagnostics

            row = self.processed_df.iloc[i]

            weight_financial = row['imprich']
            weight_social = row['ppltrst']
            weight_personal = row['ccrdprs']
            total_weight = weight_financial + weight_social + weight_personal
            weight_financial /= total_weight
            weight_social /= total_weight
            weight_personal /= total_weight

            # Choose an initial product for the agent
            initial_product = np.random.choice(self.products)

            # Creating an agent with the sampled and calculated attributes
            agent = ConsumerAgent(
                unique_id=i,
                model=self,
                budget=budget,
                preference_sustainability=row['impenv'],
                preference_conformity=row['ipfrule'],
                stability_job=row['lknemny'],
                initial_product=initial_product,
                weight_financial=weight_financial,
                weight_social=weight_social,
                weight_personal=weight_personal
            )
            temp_agents.append(agent)
            

        self.agents = temp_agents
        return self.agents
    
    def get_network(self):
        # This method will allow you to access the network graph
        return self.G

    def place_agents_on_grid(self):
        for agent in self.agents:
            node = agent.unique_id
            self.schedule.add(agent)
            self.grid.place_agent(agent, node)


    def step(self):
        random.seed(self.config['seed'])
        np.random.seed(self.config['seed'])

        # Introduce True Price products from a specific iteration
        if not self.true_price_introduced and self.schedule.steps >= 3:  # X is the step at which TP products are introduced
            self.introduce_true_price_product()
            self.true_price_introduced = True
       
         # Before agents take their steps, check how many have no product
        agents_without_products = [agent for agent in self.schedule.agents if agent.chosen_product is None]
        if agents_without_products:
            print(f"Iteration {self.schedule.steps}: {len(agents_without_products)} agents without products")

        if random.random() < self.rewiring_probability:
            self.rewire_network()

        self.schedule.step()

        agents_without_products_post = [agent for agent in self.schedule.agents if agent.chosen_product is None]
        if agents_without_products_post:
            print(f"Iteration {self.schedule.steps} post step: {len(agents_without_products_post)} agents without products")


        self.datacollector.collect(self)

        adoption_rate = self.calculate_true_price_adoption_rate()
        #print(f"Iteration {self.schedule.steps}: True Price Adoption Rate = {adoption_rate}%")
 
    def rewire_network(self):
        np.random.seed(self.config['seed'])
        random.seed(self.config['seed'])
        nodes = list(self.G.nodes())
        edges = list(self.G.edges())
        num_to_rewire = int(len(edges) * self.rewiring_probability)

        for _ in range(num_to_rewire):
            if not edges:
                break

            edge = random.choice(edges)
            new_node = random.choice(nodes)

            while new_node == edge[0] or self.G.has_edge(edge[0], new_node):
                new_node = random.choice(nodes)
                if new_node not in nodes:
                    break

            if self.G.has_edge(edge[0], edge[1]):
                self.G.remove_edge(*edge)
                self.G.add_edge(edge[0], new_node)
                edges.remove(edge)

        self.update_agents_neighbors()



    def update_agents_neighbors(self):
        for agent in self.schedule.agents:
            neighbors_nodes = self.G.neighbors(agent.unique_id)  # Changed from self.network to self.G
            agent.neighbors = [self.schedule.agents[n] for n in neighbors_nodes]



    def introduce_true_price_product(self):
        np.random.seed(self.config['seed'])
        random.seed(self.config['seed'])
        num_tp_products = int(len(self.products) * self.TP_percentage)
        
        tp_product_indices = np.random.choice(range(len(self.products)), num_tp_products, replace=False)
        
        for idx in tp_product_indices:
            product = self.products[idx]
            product.is_true_price = True
            product.tp_available = True
            min_increase = self.min_increase_percentage
            max_increase = self.max_increase_percentage
            product.price_increase_percentage = min_increase + (1 - product.green_score) * (max_increase - min_increase)
            product.normal_price *= (1 + product.price_increase_percentage / 100)

        #print(f"Introducing {num_tp_products} True Price products.")  




    def min_satisfaction(model):
        return min(agent.is_satisfied() for agent in model.schedule.agents)
   
    def max_satisfaction(model):
        return max(agent.is_satisfied() for agent in model.schedule.agents)
 
    def create_products(self):
        np.random.seed(self.config['seed'])
        random.seed(self.config['seed'])
        products = []
        num_products = self.num_products
        low_price, high_price = self.product_price_range
        mean_log_price = np.log((low_price + high_price) / 2)
        sigma = 0.5 # Adjust sigma to control the spread of the distribution

        for i in range(num_products):
            normal_price = np.random.lognormal(mean_log_price, sigma)
            # Ensure the price falls within the specified range
            while normal_price < low_price or normal_price > high_price:
                normal_price = np.random.lognormal(mean_log_price, sigma)
        
            green_score = np.random.rand()
            remediation_level = np.random.rand()
            products.append(Product(i, False, normal_price, 0, green_score, remediation_level))
    
        # Print the product costs for verification
        product_costs = [product.normal_price for product in products]
        #print("Product Costs:")
        #print(product_costs)

        return products
 
    def calculate_true_price_adoption_rate(self):
        total_agents = self.num_agents
        true_price_users = sum([1 for agent in self.schedule.agents if agent.is_using_true_price])
        return (true_price_users / total_agents) * 100
 