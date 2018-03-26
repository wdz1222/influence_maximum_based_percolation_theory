import networkx as nx

class InfluenceIndex:

    @staticmethod
    def calculate_diffusivity(n1, n2):
        t1 = n1['trust']
        t2 = n2['trust']
        d1 = n1['distrust']
        d2 = n2['distrust']
        diff_trust, diff_distrust = InfluenceIndex.calculate_diffusivity_tunple(t1, t2, d1, d2)
        diffusivity = (1-diff_trust+diff_distrust)/2
        return diffusivity

    @staticmethod
    def calculate_diffusivity_tunple(t1, t2, d1, d2):
        return t1*t2/(1+(1-t1)*(1-t2)), (d1+d2)/(1+d1*d2)
