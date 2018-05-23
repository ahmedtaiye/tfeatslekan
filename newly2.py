from __future__ import print_function
import nltk

from nltk.sem.logic import *

y=boolean_ops()
print('############################',y)
y1=equality_preds()
print('############################',y1)
z1= binding_ops()
print('############################',z1)



v = [('CEO','a1'),('VP','b1'),('DIRECTORS','c1'), ('EMPLOYEES','d1'), ('ACTIVE','a2'), ('NOT ACTIVE','a3')]
v1=[ ('Kenneth Lay',set(['a1','d1','a2'])),
('Jeff Skillings',set(['a1','d1','a2'])),
('Richard Shapiro',set(['b1', 'a3'])),
#('Richard Shapiro',set(['b1'])),
('Steven Kean',set(['b1', 'd1'])),
('Sara Shackleton',set(['b1', 'd1'])),
('James Steffes',set(['b1',  'd1', 'a3'])),
('Jeff Dasovich',set(['c1','c1','d1'])),
('Tana Jones',set(['a2','d1'])),
('Kay Mann',set(['a2','d1'])),
('Mark Taylor',set(['a2','d1'])),
('Davis Pete',set(['a2','d1'])),
('Chris G',set(['d1', 'a3'])),
('Kate Syemmes',set(['a3','d1'])),
('DIRECTORS',set(['Jeff Dasovich'])),
('VP',set(['Steven Kean','Sara Shackleton','James Steffes', 'Richard Shapiro'])),
('CEO',set(['Kenneth Lay', 'Jeff Skillings'])),
 ('POI', set([('Kenneth Lay','Jeff Skillings','Richard Shapiro','Tana Jones','Kay Mann','Mark Taylor','Chris G',
              'Kate Syemmes', 'Davis Pete','Steven Kean','Sara Shackleton','James Steffes','Jeff Dasovich')]))]
val2 = nltk.sem.Valuation(v)
val3 = nltk.sem.Valuation(v1)
print(val2)
print (val3)


from nltk.sem.logic import LogicParser
from nltk.inference import TableauProver
dom3 = val3.domain
m3 = nltk.sem.Model(dom3, val3)
g = nltk.sem.Assignment(dom3)
lpq = LogicParser()
fmla1 = lpq.parse('(POI(x) -> exists y.(CEO(y) and chase(x, y)))')
m3=m3.satisfiers(fmla1, 'x', g)
print(m3)

from nltk.sem.drt import *
dexpr = DrtExpression.fromstring
CEO_a1 = dexpr('CEO(a1)')
EMPLOYEES_d1 = dexpr('EMPLOYEES(d1)')
x = dexpr('x')
print(DRS([x], [CEO_a1, EMPLOYEES_d1]))

drs1 = dexpr('([x],[CEO(a1),EMPLOYEES(d1)])')
print(drs1)

#merge attributes to nodes
drs2 = dexpr('([y],[ACTIVE(a2),NOTACTIVE(a3)])')
drs3 = drs1 + drs2
print(drs3)
print(drs3.simplify())

#implies condition
s = '([], [(%s -> %s)])' % (drs1, drs2)
print(dexpr(s))

#The fol() method converts DRSs into FOL formulae.
print(dexpr(r'([x],[CEO(a1), EMPLOYEES(d1)])').fol())
print(dexpr(r'([x],[CEO(a1), ACTIVE(a2)])').fol())
print(drs3.pretty_format())

from nltk.parse import load_parser
from nltk.sem.drt import DrtParser

print(dexpr(r'([x,y],[works(x,y)])'))
print(dexpr(r'([x],[CEO(x), EMPLOYEES(x)])'))


