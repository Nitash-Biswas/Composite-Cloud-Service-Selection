# Dynamic Cloud Service Selection Model

## The Problem

- Imagine you need some cloud services for your next project, maybe some virtual machines (like AWS EC2, Google Compute Engine), storage services (like AWS S3, Google Cloud), Database options (MongoDB Atlas, Firebase), or AI alternatives (OpenAI, AWS Rekognition).
- **But with this vast array of choices, which ones you should use? And from which of the providers?**
- Some offer lower latencies than others (response time), some are more cost affective, some are from the major providers like Google and AWS with years of trust and millions of users behind them (more availability) where some offer more resistance to any unexpected failures.
- Often, these choices contradict each other, like if a service offers low latency, it comes at a higher cost.
- So there is need of a way to provide all these required cloud services to the user, from different providers, taking account of all these metrics (attributes) in the most effective way.

## The Idea
- You’re not just minimizing one thing (like in Travelling Salesman Problem), but balancing multiple conflicting objectives cost, latency, availability, and reliability.
- This is best seen as a `multi-objective combinatorial optimization problem.`
- This belongs to a class of problems in which finding optimal solutions can become extremely time-consuming as the problem size increases. 
- So we use `Metaheuristic Algorithms` as they provide good, but not necessarily optimal, solutions in a reasonable amount of time.
- Examples: `Ant Colony Optimization (ACO)`, `Genetic Algorithms (GA)`, `Particle Swarm Optimization (PSO)`  

## 📊 Modeling It Mathematically

Suppose you need 4 cloud services:
- Compute (e.g., EC2, GCE)
- Storage (e.g., S3, GCS)
- Database (e.g., MongoDB Atlas, Firebase)
- AI/ML (e.g., OpenAI, Rekognition)

Let:

- `S = {s₁, s₂, ..., sₙ}` be the list of candidate cloud services from different providers.
- Each service `sᵢ` has associated attributes:
  - `c(sᵢ)`: Cost
  - `l(sᵢ)`: Latency (response time)
  - `a(sᵢ)`: Availability (uptime / SLA)
  - `r(sᵢ)`: Reliability

The goal is to **select one service per required category** (compute, storage, etc.), such that the overall solution:

- Minimizes total cost
- Minimizes latency
- Maximizes availability
- Maximizes reliability

This can be formulated as a **multi-objective optimization** problem:

### 🎯 Objective Function

We define a weighted sum objective function:

minimize α * Σ c(sᵢ) + β * Σ l(sᵢ) - γ * Σ a(sᵢ) - δ * Σ r(sᵢ)

Where:
- `α`, `β`, `γ`, `δ` are tunable weights that reflect the user’s priority (e.g., cost-sensitive vs performance-sensitive).
- The summation `Σ` is over the selected services from each required category.




## Proposed Solution

A dynamic cloud service selection model powered by Ant Colony Optimization (ACO). The objectives include:

1. Enabling adaptable cloud service choices in real-time, based on multiple attributes: 
   - Cost
   - Reliability
   - Response Time
   - Availability
2. Addressing missing data effectively.
3. Ensuring scalability for large data sets.
4. Optimizing cloud service selection for improved performance and cost-effectiveness.

## WorkFlow
![Flowchart (2)](https://github.com/user-attachments/assets/6879bcb0-59b7-4f12-b13e-527d6504164f)

## Performance
The proposed approach is tested against two of the popular meta-heurisitc optimisation techniques, 
**genetic algorithm** and **bee colony algorithm** 
> Lower Lp score is better.
   
![Screenshot 2024-07-13 at 11 18 13 PM](https://github.com/user-attachments/assets/a7fd0190-94a6-4b14-a823-ad25841136c5)

## Scalability
Evaluated scalability of the ACO model by analyzing convergence time across different service path lengths (10 to 100 services) and service candidate set sizes (5 to 40), simulating cloud service providers like AWS, Google Cloud, and Azure, and covering user-requested services such as virtual machines, databases, and machine learning models.

![Screenshot 2024-09-10 at 11 16 31 PM](https://github.com/user-attachments/assets/b3b2d92d-3359-46fb-b97c-875c300819a2)


## Result
The most suitable combination of different cloud services for requested services is provided, 

![Screenshot 2024-07-14 at 12 03 10 AM](https://github.com/user-attachments/assets/d0c381de-6352-4891-9856-b6636c09866f)
> C<sub>i</sub> refers to the cloud provider selected for the i<sup>th</sup> service asked by the user.

> E.g: C1 means Cloud service provider #1 provides service #5 requested by the user.
