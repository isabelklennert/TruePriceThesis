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
    def __init__(self, unique_id, model, budget, preference_sustainability, preference_social_status, preference_conformity, stability_job, initial_product, weight_financial, weight_social, weight_personal, economic_class):
        super().__init__(unique_id, model)
        self.budget = budget
        self.preference_sustainability = preference_sustainability
        self.preference_social_status = preference_social_status
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
        self.economic_class = economic_class       
 
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
 
        if satisfied and not uncertain:
            self.repeat_action()
        elif satisfied and uncertain:
            self.imitation()
        elif not satisfied and not uncertain:
            self.deliberation()  
        elif not satisfied and uncertain:
            self.imitation()
       
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

        # Financial Satisfaction: Non-linear function, sharply decreases as cost approaches budget
        # If the product price exceeds the budget then we will have a satisfaction below 0
        # Satisfaction can range between 0 and 1
        # Therefore if it negative, by using the max and min below we ensure that the satisfaction will be 0
        self.F_satisfaction = max(0, min(1, 1 - (product_cost / self.budget)**2))
 
        # Financial Uncertainty: Influenced by inflation rate and proportion of budget spent as well as stability of job
        proportion_spent = product_cost / self.budget
        job_stability_factor = 1 - self.stability_job #1 is fully stable so then less uncertain
        inflation_impact = self.model.inflation_rate / 100 * job_stability_factor
        self.F_uncertainty = max(0, min(1, proportion_spent + job_stability_factor + inflation_impact))
 

    def calculate_social_satisfaction_and_uncertainty(self, product):
        total_agents = len(self.model.schedule.agents)
        same_product_count = sum(1 for agent in self.model.schedule.agents if agent.previous_true_price_usage == self.previous_true_price_usage)
        conformity = same_product_count / total_agents if total_agents > 0 else 0

        # Social satisfaction is now based on overall conformity to the product usage in the whole model
        self.S_satisfaction = max(0, min(1, (conformity * self.preference_conformity) ** 0.5))

        # Social uncertainty could be based on the proportion of agents that change their product choice frequently
        past_true_price_users = sum(1 for agent in self.model.schedule.agents if agent.has_used_true_price)
        previous_true_price_users = sum(1 for agent in self.model.schedule.agents if agent.previous_true_price_usage)
    
        if past_true_price_users > 0:
            stability = previous_true_price_users / past_true_price_users
            self.S_uncertainty = 1 - stability
        else:
            self.S_uncertainty = 0  # If no one has used true price, uncertainty is minimal

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
 
    def imitation(self):
        self.decision_mode = "imitate"
        product_count = {}
        tp_product_count = 0
        non_tp_product_count = 0

        # Instead of iterating over neighbors, iterate over all agents in the model
        for other_agent in self.model.schedule.agents:
            product = other_agent.last_purchased_product
            if product:
                product_count[product] = product_count.get(product, 0) + 1
                if product.is_true_price:
                    tp_product_count += 1
                else:
                    non_tp_product_count += 1
 
        if product_count:
            max_count = max(product_count.values())
            most_popular_products = [product for product, count in product_count.items() if count == max_count]

            if len(most_popular_products) > 1:
                # Tie between products, select randomly from tied products
                self.chosen_product = random.choice(most_popular_products)
            else:
                # Clear popular product
                self.chosen_product = most_popular_products[0]
        else:
            # No clear popular choice, pick randomly from all products
            self.chosen_product = random.choice(self.model.products)


 
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
        self.num_agents = config['num_agents']
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

        #KDE data
        self.kde_models = {attribute:joblib.load(os.path.join(config['kde_models_dir'], f"kde_{attribute}.pkl")) for attribute in ['ccrdprs', 'hincfel', 'impenv', 'impfree', 'imprich', 'inctxff', 'inprdsc', 'ipfrule', 'iplylfr', 'lkredcc', 'sclmeet', 'wrclmch', 'wrkctra']}
 
        #Load data
        self.income_distribution = pd.read_csv('datasets/income_distribution.csv')
 
        # Create products
        self.products = self.create_products()
        self.agents = self.create_agents()

    
        self.datacollector = DataCollector(
            model_reporters={
                #"Min_Satisfaction": lambda model: min(agent.is_satisfied() for agent in model.schedule.agents),
                #"Max_Satisfaction": lambda model: max(agent.is_satisfied() for agent in model.schedule.agents),
                #"Min_Uncertainty": lambda model: min(agent.is_uncertain() for agent in model.schedule.agents),
                #"Max_Uncertainty": lambda model: max(agent.is_uncertain() for agent in model.schedule.agents),
                "True_Price_Adoption_by_Class": lambda model: {
                    'lower': 100 * sum(1 for agent in model.schedule.agents if agent.economic_class == 'lower' and agent.is_using_true_price) / model.num_agents,
                    'middle': 100 * sum(1 for agent in model.schedule.agents if agent.economic_class == 'middle' and agent.is_using_true_price) / model.num_agents,
                    'upper': 100 * sum(1 for agent in model.schedule.agents if agent.economic_class == 'upper' and agent.is_using_true_price) / model.num_agents,
                    }
                },
            agent_reporters={
                "True_Price_Usage": "is_using_true_price",
                "Repeat": lambda a: a.decision_mode == "repeat",
                "Imitate": lambda a: a.decision_mode == "imitate",
                "Deliberate": lambda a: a.decision_mode == "deliberate",
                "Social_Compare": lambda a: a.decision_mode == "social_compare",
                "Satisfaction": lambda a: a.is_satisfied(),
                "Uncertainty": lambda a: a.is_uncertain(),
                "Economic_Class": "economic_class"
            }
        )
    

    def sample_from_kde(self, kde_model, n_samples=1):
        samples = kde_model.sample(n_samples)
        return samples.reshape(-1)  
 
    def create_agents(self):
        temp_agents = []
        
        incomes = self.income_distribution['Income in thousands'].values
        probabilities = self.income_distribution['Probability'].values
        lower_threshold = np.percentile(incomes, 30)
        upper_threshold = np.percentile(incomes, 70)

        for i in range(self.num_agents):
            # Sampling budget from the income distribution
            annual_budget = np.random.choice(self.income_distribution['Income in thousands'].values, p=self.income_distribution['Probability'].values)
            budget = annual_budget / 52 #weekly budget

            # Assign class based on budget
            if annual_budget < lower_threshold:
                economic_class = 'lower'
            elif annual_budget > upper_threshold:
                economic_class = 'upper'
            else:
                economic_class = 'middle'

            # Sampling preferences and weights using KDE
            preference_sustainability_samples = self.sample_from_kde(self.kde_models['ccrdprs'], 1)
            preference_sustainability = preference_sustainability_samples[0]

            preference_conformity_samples = self.sample_from_kde(self.kde_models['iplylfr'], 1)
            preference_conformity = preference_conformity_samples[0]

            stability_job_samples = self.sample_from_kde(self.kde_models['wrkctra'], 1)
            stability_job = stability_job_samples[0]

            weight_financial_samples = self.sample_from_kde(self.kde_models['imprich'], 1)
            weight_financial = weight_financial_samples[0]

            weight_social_samples = self.sample_from_kde(self.kde_models['sclmeet'], 1)
            weight_social = weight_social_samples[0]

            weight_personal_samples = self.sample_from_kde(self.kde_models['impenv'], 1)
            weight_personal = weight_personal_samples[0]

            # Normalize weights so they sum up to 1
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
                preference_sustainability=preference_sustainability,
                preference_social_status=np.random.uniform(0, 1),
                preference_conformity=preference_conformity,
                stability_job=stability_job,
                initial_product=initial_product,
                weight_financial=weight_financial,
                weight_social=weight_social,
                weight_personal=weight_personal,
                economic_class=economic_class 
            )
            temp_agents.append(agent)
            self.schedule.add(agent)


            #print(f"Created agent {i}: Budget={budget}, Initial Product ID={initial_product.product_id}")

        self.agents = temp_agents
        return self.agents
    


    def step(self):
   
        # Introduce True Price products from a specific iteration
        if not self.true_price_introduced and self.schedule.steps >= 3:  
            self.introduce_true_price_product()
            self.true_price_introduced = True
       
        # Before agents take their steps, check how many have no product
        agents_without_products = [agent for agent in self.schedule.agents if agent.chosen_product is None]
        if agents_without_products:
            print(f"Iteration {self.schedule.steps}: {len(agents_without_products)} agents without products")

        self.schedule.step()

        agents_without_products_post = [agent for agent in self.schedule.agents if agent.chosen_product is None]
        if agents_without_products_post:
            print(f"Iteration {self.schedule.steps} post step: {len(agents_without_products_post)} agents without products")

        self.datacollector.collect(self)

        adoption_rate = self.calculate_true_price_adoption_rate()
        #print(f"Iteration {self.schedule.steps}: True Price Adoption Rate = {adoption_rate}%")
 
    def introduce_true_price_product(self):
        num_tp_products = int(len(self.products) * self.TP_percentage)
        tp_product_indices = np.random.choice(range(len(self.products)), num_tp_products, replace=False)
        
        for idx in tp_product_indices:
            product = self.products[idx]
            product.is_true_price = True
            product.tp_available = True
            min_increase = self.min_increase_percentage # Minimum price increase for the highest green score
            max_increase = self.min_increase_percentage  # Maximum price increase for the lowest green score
            product.price_increase_percentage = min_increase + (1 - product.green_score) * (max_increase - min_increase)
            product.normal_price *= (1 + product.price_increase_percentage / 100)

        #print(f"Introducing {num_tp_products} True Price products.")  
        for agent in self.schedule.agents:
            agent.deliberation()

    def min_satisfaction(model):
        return min(agent.is_satisfied() for agent in model.schedule.agents)
   
    def max_satisfaction(model):
        return max(agent.is_satisfied() for agent in model.schedule.agents)
 
    def create_products(self):
        products = []
        num_products = self.num_products
        low_price, high_price = self.product_price_range
        for i in range(num_products):
            normal_price = np.random.normal(low_price, high_price)
            green_score = np.random.rand()
            remediation_level = np.random.rand()
            products.append(Product(i, False, normal_price, 0, green_score, remediation_level))  
        return products
 
    def calculate_true_price_adoption_rate(self):
        total_agents = self.num_agents
        true_price_users = sum([1 for agent in self.schedule.agents if agent.is_using_true_price])
        return (true_price_users / total_agents) * 100
 