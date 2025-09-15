from soc_emp import Dynamics

if __name__ == '__main__':
    ## simulation horizon
    horizon = 50
    max_power = 1.0
    steps = 1500

    ## load in xml
    xml_path = 'xml/custom/pendulum.xml'
    dyn = Dynamics(path = xml_path, dt = 0.01)

    print(dyn)