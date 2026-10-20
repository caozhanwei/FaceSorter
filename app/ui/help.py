# -*- coding: utf-8 -*-
"""使用帮助：help_text 为纯函数（可测），HelpDialog 负责展示。"""
from PyQt6.QtWidgets import QDialog, QTextBrowser, QVBoxLayout


def help_text() -> str:
    return """【快速上手】
1. Ctrl+O 选择照片根目录（可加多个，子文件夹会递归扫描）
2. F5 扫描：检测人脸并自动按人聚类；线程数在 设置→扫描线程 可调（多线程更快）
3. 左栏点人物看 TA 的照片堆；右键人物可重命名/合并/置顶
4. 「未归类脸」里选中一张或多张脸，拖拽到左栏人物即归类；右键可新建人物
5. 「全部脸」是全库清晰脸列表（质量降序）：双击直接进该人物的照片堆
6. 满意后选中人物 Ctrl+E 导出副本（原文件永不改动）

【看大图（双击照片）】
- 当前人物的脸 = 绿框，其他人 = 灰框
- ←/→ 翻页；滚轮或 +/- 缩放；按住拖动平移；F 适应窗口；Esc 关闭

【调聚类效果】
- 人物太杂（一人分多人）→ 设置里调低阈值；脸混人（多人合一人）→ 调高阈值
- 改阈值立即自动重排（用已存特征，不重扫照片，几秒完成）
- 时间加权可让「相近时间拍的更像」权重更高
- 手动命名/拖拽过的归属在重排时全部保留，不会被自动结果覆盖

【小提示】
- 增量扫描：只处理新增/变更的照片，重复 F5 很快
- 照片文件、EXIF、时间戳完全离线处理，不上传任何数据
- 阈值、时间加权、线程数都在 设置（Ctrl+,）里
- 随时按 F1 回到本帮助"""


class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("使用帮助")
        self.resize(560, 520)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self._browser = QTextBrowser()
        self._browser.setPlainText(help_text())
        lay.addWidget(self._browser)
