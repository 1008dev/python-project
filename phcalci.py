import streamlit as st
import pandas as pd
import numpy as np
from sympy import *
import sympy as sp
from scipy.integrate import odeint 
from sympy.interactive import printing
printing.init_printing(use_latex=True)
from PIL import Image
import matplotlib.pyplot as plt

st.title(' Harmonic Oscillators ')

option1 = st.selectbox('select the type of oscillation : ',
                       (' none selected ',' free oscillation ',' free damped oscillation ',' forced oscillation ' ))

if option1 == ' free oscillation ':
    option = st.radio('Select the system :',
                  ('none selected',' Spring-mass System ',' Simple Pendulum ',' Compound Pendulum ',' Electrical System '))

    
    if option == ' Spring-mass System ':
        col1, col2 = st.columns(2)

        with col1:
            file_path= 'https://upload.wikimedia.org/wikipedia/commons/2/25/Animated-mass-spring.gif'
            st.image(file_path, width=150)

        with col2:
            st.header('spring mass system : ')
                
            t = sp.symbols('t')
            x= sp.Function('x')(t)
            k= st.number_input('Enter the value of spring constant :')
            m= st.number_input(' Enter the mass of the object attched to spring: ')
            eq= sp.Eq(x.diff(t,2),-(k/m)*x)
            st.write('The differntial equation of spring mass system is :') 
            st.latex(sp.latex(eq))

            T=2*3.142*np.sqrt(m/k)
            st.write(' The time period of oscillation is : ',T)

            omega=np.sqrt(k/m)
            st.write(' Its angular frequency is : ',omega)

        

    
    elif option == ' Simple Pendulum ':
        col1, col2 = st.columns(2)

        with col1:
            file_path= 'https://upload.wikimedia.org/wikipedia/commons/6/6f/Pendulum-no-text.gif'
            st.image(file_path, width=350)

        with col2:
            st.header('Simple pendulum : ')
            g = st.number_input('Enter the value of acceleration due to gravity (g):')
            st.write('g = ', g)

            l = st.number_input('Enter the value of length of the string in meter :')
            st.write('l = ', l)

            m = st.number_input('Enter the value of mass of the object attached to massless string (in kg) :')
            st.write('m = ', m)

            t = sp.symbols('t')
            x= sp.Function('x')(t)
            eq= sp.Eq(x.diff(t,2),-(g/l)*x)
            st.write('The differntial equation of simple pendulum is :') 
            st.latex(sp.latex(eq))

            T=2*3.142*np.sqrt(l/g)
            st.write(' The time period of oscillation is : ',T)

            omega=np.sqrt(g/l)
            st.write(' Its angular frequency is : ',omega)



        def simple_pendulum(theta0, omega0, tmax, dt, g, L):
            """
            Solve the differential equation of a simple pendulum numerically using odeint.
    
            Parameters:
            theta0 (float): initial angle in radians
            omega0 (float): initial angular velocity in radians per second
            tmax (float): maximum time in seconds
            dt (float): time step in seconds
            g (float): acceleration due to gravity in meters per second squared
            L (float): length of the pendulum in meters
    
            Returns:
            tuple: two arrays containing the time and the angular position of the pendulum at each time step
            """
            def f(y, t, g, L): 
                theta, omega = y
                dtheta_dt = omega
                domega_dt = -(g/L)*np.sin(theta)
                return [dtheta_dt, domega_dt]
    
            # Create time array
            t = np.arange(0, tmax+dt, dt)
    
            # Create initial conditions
            y0 = [theta0, omega0]
    
            # Solve differential equation numerically
            sol = odeint(f, y0, t, args=(g, L))
     
            # Extract angular position from solution
            theta = sol[:, 0]
    
            # Return time and angular position arrays
            return t, theta

        # Set default values
        theta0_default = 0.1
        omega0_default = 0
        tmax_default = 10
        dt_default = 0.01
        g_default = 9.81
        L_default = 1

        # Define input widgets
        theta0 = st.sidebar.slider("Initial angle (radians)", -np.pi, np.pi, theta0_default)
        omega0 = st.sidebar.slider("Initial angular velocity (radians/s)", -10, 10, omega0_default)
        tmax = st.sidebar.slider("Maximum time (s)", 1, 30, tmax_default)
        dt = st.sidebar.slider("Time step (s)", 0.01, 0.1, dt_default, step=0.01)
        g = st.sidebar.number_input("Acceleration due to gravity (m/s^2)", value=g_default)
        L = st.sidebar.number_input("Pendulum length (m)", value=L_default)

        # Solve differential equation
        t, theta = simple_pendulum(theta0, omega0, tmax, dt, g, L)

        # Plot results
        fig, ax = plt.subplots()
        ax.plot(t, theta)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angular position (radians)")
        st.pyplot(fig)

    
    elif option == ' Compound Pendulum ':
        col1, col2 = st.columns(2)

        with col1:
            image = Image.open('compoundpendulum.jpg')
            st.image(image)

        with col2:
            st.header('Compound Pendulum: ')
            I = st.number_input('Moment of inertia of the pendulum :')
            st.write('I = ', I)

            l = st.number_input('Enter the length of the pendulum :')
            st.write('l = ', l)

            m = st.number_input('Enter the value of mass of the object (in kg) :')
            st.write('m = ', m)

            g = st.number_input('Enter the value of acceleration due to gravity (g) :')
            st.write('g = ', g)

            t = sp.symbols('t')
            x= sp.Function('x')(t)
            eq= sp.Eq(x.diff(t,2),-(m*g*l/I)*x)
            st.write('The differntial equation of compound pendulum is :') 
            st.latex(sp.latex(eq))

            T=2*3.142*np.sqrt(I/m*g*l)
            st.write(' The time period of oscillation is : ',T)

            omega= 1/ np.sqrt(m*g*l/I)
            st.write(' Its angular frequency is : ',omega)


    elif option == ' Electrical System ':
        
        col1, col2 = st.columns(2)

        with col1:
            file_path= 'https://upload.wikimedia.org/wikipedia/commons/1/1d/Tuned_circuit_animation_3.gif'
            st.image(file_path, width=300)

        with col2:
            st.header('The Electrical System : ')
            L = st.number_input('Enter the value of inductance(L):')
            st.write('L = ', L)

            C = st.number_input('Enter the value of capacitance(C) :')
            st.write('C = ', C)


            t = sp.symbols('t')
            q= sp.Function('q')(t)
            eq= sp.Eq(q.diff(t,2),-(1/L*C)*q)
            st.write('The differntial equation of the electrical system is :') 
            st.latex(sp.latex(eq))

            T=2*3.142*np.sqrt(L*C)
            st.write(' The time period of oscillation is : ',T)

            omega= 1/ np.sqrt(L*C)
            st.write(' Its angular frequency is : ',omega)


 
    else:
        print('Please choose from the options ')

elif option1 == ' free damped oscillation ':
    
    st.header(' Damped oscillation : ')
    file_path= 'https://upload.wikimedia.org/wikipedia/commons/f/fa/Spring-mass_under-damped.gif'
    st.image(file_path, width=500)
    def damped_oscillator(y, t, b, k):
        dydt = [y[1], -b*y[1] - k*y[0]]
        return dydt

    def main():
        
        b = st.slider("Damping Coefficient", 0.0, 1.0, 0.1)
        k = st.slider("Spring Constant", 0.0, 10.0, 1.0)
        y0 = [1.0, 0.0]
        t = np.linspace(0, 10, 101)
        sol = odeint(damped_oscillator, y0, t, args=(b, k))
        fig, ax = plt.subplots()
        ax.plot(t, sol[:, 0], 'b', label='displacement')
        ax.plot(t, sol[:, 1], 'g', label='velocity')
        ax.legend(loc='best')
        ax.set_xlabel('time')
        ax.grid()
        st.pyplot(fig)

    if __name__ == "__main__":
        main()



elif option1 == ' forced oscillation ':
    
    st.title("Forced Oscillation Solver")

    # Define the differential equation
    def forced_oscillator(y, t, b, w0, F):
        dydt = [y[1], -b*y[1] - w0**2*y[0] + F*np.cos(w0*t)]
        return dydt
    
    # Define the app
    def main():
        # Define the parameters
        b = st.slider("Damping coefficient", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        w0 = st.slider("Natural frequency", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        F = st.slider("Driving force amplitude", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

        # Define the initial conditions
        y0 = [1.0, 0.0]  # initial displacement and velocity

        # Define the time array
        t = np.linspace(0, 10*np.pi, 101)

        # Solve the differential equation
        sol = odeint(forced_oscillator, y0, t, args=(b, w0, F))

        # Plot the results
        fig, ax = plt.subplots()
        ax.plot(t, sol[:, 0], 'b', label='displacement')
        ax.plot(t, sol[:, 1], 'g', label='velocity')
        ax.legend(loc='best')
        ax.set_xlabel('time')
        ax.set_ylabel('y')
        ax.grid()

        # Show the plot
        st.pyplot(fig)

    if __name__ == "__main__":
         main()

    
