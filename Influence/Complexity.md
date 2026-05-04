# Complexity 
---
* The essence of code quality is controlling complexity.
* 代码质量的本质，是控制复杂度。
---
* Complexity comes from boundary failure. Software design is the art of defining boundaries. Refactoring is the art of repairing boundaries.
* 复杂度来自边界失效。软件设计，本质是在设计边界。重构，本质是在修复边界。
---
* The purpose of controlling complexity is to allow a finite human mind to continuously manage an ever-growing system.
* 控制复杂度的目的，是让有限的人脑，持续掌控不断增长的系统。

---

* What is the problem of complexity?
    * the tax you pay every time you read or change the code
    * cognitive overhead
    * change amplification
* Where does complexity come from?
    * layer separation failure: context + policy + mechanism are collapsed into the same control flow
    * dimension separation failure: multiple orthogonal variants are collapsed into the same function
* How to reduce complexity?
    * encapsulation: keep internal details within boundaries, reduce external cognitive load, limit detail leakage
    * separation of concerns: split different dimensions of problems, limit logical coupling propagation

