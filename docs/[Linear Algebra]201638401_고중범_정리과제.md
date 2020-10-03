# Linear Algebra 1장

Linear algebra는 숫자로 이루어진 대용량의 데이터를 vector, matrix, tensor를 이용하여 효과적으로 처리하기 위한 계산을 돕는 학문이다. 

## 1.1 Systems of Linear equations

#### Linear equation

$$
a_1x_1 + a_2x_2 + ... + a_nx_n = b \\
$$

위와 같은 형태의 방정식을 **Linear equation**이라고 한다. **a들을 coefficient**라고 하고 **x들이 unknown/variable**이고, 이 Linear equation의 해를 Solution set이라고 한다. System of Linear equation은 **(1)해가 없거나, (2)해가 오직 하나만 존재하거나, (3)해가 무수히 존재한다.** 이때 해가 존재하는 (2), (3)의 경우를 **Consistent**라고 표현하고, 해가 없는 (1)의 경우에 **Inconsistent**라고 표현한다. 그리고 만약 두 Linear equation이 같은 Solution set을 가진다면 두 Linear system이 **Equivalent**라고 표현한다.



#### Matrix Notation

$$
\begin{align*}
x_1-2x_2+x_3 &= 0 \\
2x_2-8x_3 &= 8 \\
-4x_1+5x_2+9x_3 &= -9 \\
\end{align*}
$$

위 식에 대해서 Matrix는 Column vector의 집합이라고 볼 수 있는데, 각각의 Column vector는 위 3개의 식에서 특정 Variable의 coefficient의 정보를 담고있다. 위 식의 coefficient들을 Matrix로 표현하면 아래와 같다.
$$
\begin{bmatrix}
1 & -2 & 1 \\
0 & 2 & -8 \\
-4 & 5 & 9 
\end{bmatrix}  
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \begin{bmatrix}
1 & -2 & 1 &0\\
0 & 2 & -8 &8\\
-4 & 5 & 9 &-9
\end{bmatrix} \tag{4}
$$
왼쪽 Matrix와 같이 각 Variable에 대한 Coefficient들을 Matrix으로 표현한 것을 Coefficient matrix라고 하고, 오른쪽과 같이 등호 옆 Constant까지 더한것을 Augmented matrix라고 한다.



#### Elementary Row Operations

복잡한 Linear system을 간단하게 만들어 쉽게 풀기위한 방법이다. 아래 세가지 방법을 혼합하여 해는 그대로 유지한 상태에서 Linear system을 간단하게 만들 수 있다. **그리고 Row operation은 reversible하다.**

- **Replacement**: 한 row를 해당 row와, Scaling된 어떤 row의 sum으로 대체한다.
- **Interchange**: 두 row의 위치를 서로 바꾼다.
- **Scaling**: Row를 0이 아닌 수로 곱한다.

한 Matrix가 Elementary row operation을 통해 다른 Matrix로 Transform될 수 있다면, 그때 두 Matrix는 **Row Equivalent**하다고 표현한다. 



## 1.2 Row Reduction and Echelon Forms

#### Definition of Echelon Form

우선 두가지 단어를 짚고 넘어가자. Nonzero row or column은 0이 아닌 element를 적어도 하나 가지고 있는 row나 column을 의미한다. Leading entry는 nonzero row에서 0이 아니면서 가장 왼쪽에 있는 최초 element를 의미한다. Echelon form은 사다리꼴을 의미하는데, Matrix가 아래 세가지 조건을 만족하면 Echelon form이라고 한다.

- Non-zero row들은 all-zero row들보다 위에 있어야한다.
- 각각의 Row의 Leading entry들은 자기보다 위에 있는 row의 Leading entry보다 오른쪽 열에 있어야한다.
- Leading entry 아래있는 column의 성분들은 모두 0이어야 한다.

아래의 추가적인 조건을 더 만족할 경우, Reduced Echelon Form이라고 하고, 서로 Row Equivalent한 Matrix는 서로 같은, 오직 하나의 Unique한 Reduced Echelon matrix를 가진다. 

- 각각의 nonzero row의 leading entry는 전부 1이다.
- Leading 1을 포함하는 각각의 Column의 다른 entry는 전부 0이다.

<br>

#### Pivot Position

Reduced Echelon form이 항상 unique하니까 leading entries도 항상 같은 위치다. Pivot position이란 Reduced echelon form에서 leading 1의 위치를 의미한다. 그리고 Pivot column은 Pivot position을 포함하는 column을 의미한다.
$$
A=
\begin{bmatrix}
0 & -3 & -6 & 4 & 9 \\
-1 & -2 & -1 & 3 & 1 \\
-2 & -3 & 0 & 3 & -1 \\
1 & 4 & 5 & -9 & -7
\end{bmatrix} \ \ \
$$
위 A와 같은 Matrix가 있을 경우, Row operations을 이용해서 Reduced Echelon form으로 만들면서 Pivot position에 어떤 수가 있는 지를 알아낼 수 있다. 이렇게 알아낸 Pivot position과 column을 통해 변수들이 결정되는 부분에 개입하고, Linear equation을 푸는데 있어서 핵심적인 element들의 위치를 알 수 있다.

<br>

#### Solutions of Linear Systems

$$
\begin{bmatrix}
1 & 0 & -5 & 1 \\
0 & 1 & 1 & 4 \\
0 &0 &0 &0 
\end{bmatrix} \ \ \ \ \ \ \ \ \ \ 
\begin{align*}
x_1 - 5x_3 & = 1 \\
x_2 +x_3 &= 4 \\
0 &= 0
\end{align*}
$$

위 Augmented matrix를 연립방정식으로 옮기면 오른쪽과 같다. 이때 Pivot column에 해당하는 x_1과 x_2를 Basic Variable이라고 부르고, x_3을 Free variable이라고 부른다. Free variable이라는건 어떤 수가 선택되던지 자유라는 뜻이고, Free variable의 값에 의해서 x_1과 x_2같은 Basic variable의 값이 정해진다. 

<br>

#### Parametric Descriptions of Solution Sets

위에서의 식처럼, Free variable을 Paramter로 사용하여 Solution set을 나타낼 수 있다. 어떤 Linear system이 Consistent 하고 Free variable을 가지고있는 경우, Solution set은 많은 Parametric description을 가진다. 해가 없는 경우는 Solution set도 없고, Parametric description도 존재하지 않는다.

<br>

#### Existence and Uniqueness Question

어떤 System이 Echelon form에서 0 = b와 같은 equation이 있고, b가 0이 아니라면, 해는 없다. 하지만 저것과 같은 equation이 없을 경우에는 그 System은 Consistent하다. Free variable이 없다면 Solution은 Unique할 것이고, Free variable이 있다면 Solution은 무수히 많다.  

<br>

## 1.3 Vector Equations

Vector는 Ordered list of numbers다. index(순서)를 가지고 있고, 그 순서가 중요하다는걸 기억하자. Vector와 matrix가 Linear equation에서 표현이 되고, 해를 가진다는 조건을 어떻게 만들어낼 수 있는가에 대해서 주목하며 이 단원을 공부해야한다.

<br>

#### Vectors in $\mathbb{R^2}$

Matrix에서 하나의 Column을 뜯어냈을 때, 이것을 Column vector라고 하고, 앞으로 Vector라고 하면 Column vector를 의미한다. (분야별로 다를 수 있다.) **R^2**에서 R는 Real number를 뜻하고, ^2는 전체 Entries가 2개임을 뜻한다.

- 두개의 vector가 R^2에 속할때, corresponding하는 위치에 있었던 entries가 같을 경우에 "Two vector are equal."

- **u**와 **v**라는 vector가 있을 때, **u**하고 **v**의 **sum**은 corresponding하는entries 끼리 더한 결과다.

- real number c와 vector **u**가 있을때 , **scalar multiple**은 하나의 scalar인 c를 모든 element에 다 곱해준 것이다.

연산에 대해서 닫혀있기 때문에 결과값도 real number의 집합에 포함된다.

<br>

#### Geometric Descriptions of R Squared

vector space 자체가 어떤 공간이니까 기하학적으로 모델링할 수는 있지만, 3차원초과해서 넘어가게 되면 머리속에 그려지지가 않는다. 하지만 개념은 그대로 확장될 수 있다.
$$
Geometric \ point \ (a,b) = Column \ vector \begin{bmatrix} a \\ b \end{bmatrix}
$$
그럼 이대로 본다면, R^2는 하나의 무한히 넓은 Plane이라고 할 수 있고, 여기에 점을 하나 딱 찍으면 그게 Vector라고 할 수 있다.

<br>

#### Linear Combination

Vector **v_1, v_2, v_3, ..., v_p**가 있고, Scalar c_1, c_2, c_3, ..., c_p가 있을 때 Vector y는 아래와 같이 정의되고 이를 c를 weight로 한 **v_1~v_p**의 Linear combination이라고 한다.
$$
\textbf{y}=c_1\textbf{v}_1+\cdots+c_p \textbf{v}_p
$$

#### Span

span{v_1, v_2,... , v_p}는 아래와 같이 나타낼 수 있는 Vector의 모든 집합이다.
$$
c_1 \textbf{v}_1 +c_2 \textbf{v}_2 +\cdots + c_p \textbf{v}_p
$$
<br>

#### A Geometric Description of Span{v}

v가 만약 R^n의 nonzero vector라고 할 때, Span{v}는 Geometric적으로 v에 scalar multiple을 한 것의 집합이다. 즉, 한 직선이 있다고 할 때, 위아래로의 연장선을 포함한 것이고 볼 수 있다. 

만약 두개의 Vector u,v가 있을 때, Span{u, v}를 생각해보면, u와 v는 평면을 이루고 있을 것이고, (nonzero라면) Span{u, v}는 이 평면이 연장된 형태일 것이다.

<div>
  <img src="https://dsaint31x.github.io/ds_gitbook/posts/LinearAlgebra/fig/la_01_03_10.png" width=350 height=300 border="1">
  <img src="https://dsaint31x.github.io/ds_gitbook/posts/LinearAlgebra/fig/la_01_03_11.png" width=350 height=300 border="1">
</div>

<br>

## 1.4 Matrix Equation

Vector와 Linear combination를 Matrix와 Vector로 바라봐야한다. 그럼으로써 기계가 학습하기 더 좋은 방향으로 식을 유도할 수 있다.

#### Definition

$$
A\textbf{x} =
\begin{bmatrix}
 & & \\
\textbf{a}_1 & \cdots & \textbf{a}_n\\
 & & \\
\end{bmatrix}
\begin{bmatrix}
x_1 \\ 
\vdots \\
x_n
\end{bmatrix}
=
 x_1 \textbf{a}_1  + x_2 \textbf{a}_2 + \cdots +  x_n \textbf{a}_n
$$

A가 a_1~a_n열을 가진 m x n Matrix, x가 R^n에 있다면 (행이 n개라면), A와 x의 곱은 A의 Column을 weight로 사용하는 x로 구성된 Linear combination이다.
$$
\begin{align}
x_1+2x_2-x_3 &=4 \\
-5x_2 +3 x_3 &=1 
\end{align} \ \ \ \ \ \ \ \ \ \ 
x_1 \begin{bmatrix} 1 \\ 0 \end{bmatrix} 
+x_2 \begin{bmatrix} 2 \\ -5 \end{bmatrix} 
+x_3 \begin{bmatrix} -1 \\ 3 \end{bmatrix} 
= \begin{bmatrix} 4 \\ 1 \end{bmatrix}  \ \ \ \ \ \  \ \ \ \ 
\begin{bmatrix} 1 & 2 & -1 \\ 0 & -5 & 3 \end{bmatrix} 
\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} =
\begin{bmatrix} 4 \\ 1 \end{bmatrix}
$$
왼쪽과 같은 System of linear equation을 오른쪽과 같은 Matrix * vector = b 와 같은 Form으로 정리할 수 있고, 이걸 Matrix equation이라고 부른다.

<br>

#### Theorem 4

m x n Matrix인 A에 대해서 4가지 명제가 있는데, 하나가 성립하면 나머지 모두 성립한다. (Coefficient Matrix에 대해서만, augmented matrix에선 아니다.)

1. For each b in R^m, the equation Ax has a solution
2. R^m에 속하는 b는 linear combination으로 만들어질 수 있다.

3. A의 column을 가지고 Linear combination해서 다 만들어낼 수 있다. (Span)
4. A의 모든 row에 pivot position이 있다

<br>

#### Identity Matrix

Square matrix이고, Diagonal element가 전부 1이고, 아닌곳은 전부 0일때, Identity Marix라고 한다. (I로 표현된다.) 

<br>

#### Theorem 5

A가 m x n Matrix이고, u,v는 R^n에 속하고 c가 scalar일 때, 아래 두가지가 성립한다.
$$
1. \ A(\textbf{u} +\bf{v}) = A\bf{u} +A\bf{v} \\
2. \ A(c\textbf{u}) = c(A\textbf{u})
$$
(1)은 Additivity, (2)는 Homogeneity

<br>

## 1.5 Solution Sets of Linear Systems

Solution set을 가지고있다는건 식를 만족시키는 집합을 가지고 있다는 뜻이다. 이전에는 scalar 하나하나에 집중했다면, 이제는 하나의 vector로 인식하고 geometric하게 생각할 수 있도록 Vector notation이란걸 사용한다.

<br>

#### Homogeneous Linear Systems

$$
A\vec{x} = \vec{0}
$$

식의 Right side가 0인 경우를 의미한다. 이 경우에는 해가 적어도 하나는 존재한다. (x=0) 이와같은 Zero solution을 **Trivial solution**이라고 한다. Homogeneous Linear system에서 어차피 Solution은 적어도 하나는 존재하기 때문에, 중요한건 그 외의 Solution이다. 이를 **Nontrivial solution**이라고 한다. 
$$
A\vec{x} = \vec{b}
$$


반대로 Right side가 0이 아닌 경우, 이를 Nonhomogeneous Linear System이라고 한다. 이게 만약 Consistent하다고 가정하면, 그 Solution set안에는 Homogeneous solution이 있을 것이고, 결국 여기에 어떠한 Vector를 하나 더해주면 그게 바로 Non homogeneous Linear system의 전체 Solution이 된다. 

![img](http://pds26.egloos.com/pds/201807/15/89/a0322389_5b4a31db9383e.png)

위 그림과 같이 설명하면 Homogeneous linear system에서의 solution에서 더해진 b에 의해 Translate된 것이 Non homogeneous linear system의 해가 되는 것이다.

<br>

## 1.7 Linear Independence

만약 어떤 Vector equation에서 오직 Trivial solution만 존재한다면, 그 vector들은 Linearly Independent하다. 독립관계를 알아내는게 중요한 이유는 만약 Span{v_1, v_2}가 있다고 했을 때, 둘이 Linearly dependence하다면, 그 말은 즉, 겹치는 값이 있을 수도 있다는 말이 된다. 이때 독립관계를 알아내서 가능한 적은 수의 vector로 같은 값을 만들 수 있다면 그것이 훨씬 효율적이기 때문에 독립관계를 알아내는것은 중요하다. 

> Linear combination으로 어떤 vector를 다른 vector와 똑같이 만들 수 있다면 Linear dependent, 없다면 Linear Independent다.

- Zero vector가 존재 -> Non trivial solution을 만들 수 있다. -> Linearly dependent

- Free variable 존재 -> Linear dependent -> 해가 무수히 많으니까

머신러닝에서 봤을 때 이러한 Linear dependent는 결국 같은 결과는 내는것들이 겹치게 적용되는 일이 벌어질 수 있기 때문에 Linearly Independent 관계를 가지고 있는 Feature를 모은다. (Feature들이 Unique한 특성을 반영하도록 만든다.) 

<br>

## 1.8 Introduction to Linear Transformations

Transformation(=function/mapping)

n차원 실수로 이루어진 Vector space에서, m차원으로 Mapping을 떠주는 규칙, 함수를 Transformation이라고 한다. 그리고 이게 Linear한 성질을 가지면 Linear Transformation이라고 부른다. 

- 거울을 본다고했을때 3차원 공간이 거울이라는 2차원공간에 매핑된건데 이때 거울에 있는걸 상(Image)라고 한다.

- Domain(정의역), Codomain(공역), Range(치역)

<br>

#### Matrix Transformation

- Matrix가 Transformation되면 Matrix Transformation이고, 선형적인 특성을 가지고 있다.

- 벡터와 matrix의 곱을 전개해보면 Linear combination 형식이고, 결국 b(Codomain)을 구할 수 있게된다.
- Span{a}는 실제적으로 Range of T가 된다.



#### Definition: A transformation (or Mapping) T is linear if

- Additivity: 둘을 더해서 Transformation하든, 둘다 Transformation하고 더하든 같은 결과가 나와야한다.
- Homogeneneity: scalar multiple을 먼저하고 tranformation하든, Transformation하고 scalar multiple을 하든 같은 결과가 나와야한다.

위의 두가지 속성을 만족하면 Linear Transformation이다. 그리고 이 두가지를 만족한다면 Matrix Transformation으로 표현이 가능하다.

<br>

## 1.9 The Matrix of a Linear Transformation

#### Definition: onto (전사)

![Fig3](https://dsaint31x.github.io/ds_gitbook/posts/LinearAlgebra/fig/la_01_09_03.png)

여기서 **전사**란 모든 Codomain이 domain과 전부 mapping이 되어었을때를 onto관계라고도 이야기한다. **R^m**에 속한 모든 벡터들이 domain에 있는 녀석들하고 **최소한 하나이상** 매핑된 경우이다.

<br>

#### Definition: one to one

![Fig4](https://dsaint31x.github.io/ds_gitbook/posts/LinearAlgebra/fig/la_01_09_04.png)

onto와는 다르게 하나 이상이 아니라 각각의 domain과 codomain이 **최대 하나씩**이랑만 매핑되있는 경우이다.

