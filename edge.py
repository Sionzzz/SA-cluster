#-*- encoding:utf-8 -*-
class edge(object):
    #保证node1小于node2

    def __init__(self,node1,node2):
        node1=int(node1)
        node2=int(node2)
        if node1 > node2:
            self.node1,self.node2 = node2,node1
        elif (node1 < node2):
            self.node1,self.node2 = node1,node2
        self.core = False
        self.classfied=False
        self.visited=False
        self.IsOutlier=False
        self.id=None
        self.label={}
        self.border=False

        self.weight = 1

    def getNode1(self):
        return self.node1
    def getNode2(self):
        return self.node2
    def getNodes(self):
        return self.node1,self.node2

    def __eq__(self, other):

        if self._node1!=other.node1:
            return False
        if self.node2!=other.node2:
            return False
        return True

    def __repr__(self):
        type('str')
        return '<edge %r -- %r>' % (self.node1, self.node2)

    # #重载运算符函数
    # def __ne__(self, other):
    #     return (not self.__eq__(other))
    #
    # def __hash__(self):
    #     return hash(self._node1) + hash(self._node2)


if __name__=="__main__":
   E = edge(1,2)
   # print E.getNode1
   print(E)