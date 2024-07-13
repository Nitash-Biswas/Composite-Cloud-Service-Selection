# Dynamic Cloud Service Selection Model

## Summary

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

## Result
The most suitable combination of different cloud services for requested services is provided, 

![Screenshot 2024-07-14 at 12 03 10 AM](https://github.com/user-attachments/assets/d0c381de-6352-4891-9856-b6636c09866f)
> C<sub>i</sub> refers to the cloud provider selected for the i<sup>th</sup> service asked by the user.

> E.g: C1 means Cloud service provider #1 provides service #5 requested by the user.
