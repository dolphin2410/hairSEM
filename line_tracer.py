#    hairSEM aims to quantify hair damage using a SEM image
#    Copyright (C) 2024 dolphin2410
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import ml_model
from renderer import ImageRenderer, RenderTasks
from input_manager import HairSEMEvents, InputManager, SubscriptionType
import geometrics

class LineTracer:
    """
    하나의 선을 담당하는 클래스
    """

    def __init__(self, renderer: ImageRenderer, input_manager: InputManager):
        self.renderer = renderer
        self.input_manager = input_manager
        self.start_point = None
        self.end_point = None
        self.lock = False

    def initialize(self):
        """
        초기화를 진행한다 - 좌클릭 콜백을 등록한다
        """

        self.subscription_id = self.input_manager.subscribe(SubscriptionType.LEFT_CLICK, self.on_click)

    def update(self):
        """
        업데이트를 수행한다 - 선분을 그린다
        """

        # 만약 첫번째 점은 결정되었지만 두번째 점은 결정되지 않았다면, 현재 마우스의 위치를 임시 위치로 설정한다
        if not self.lock and self.start_point is not None:
            self.end_point = self.input_manager.cursor_pos

        self.render_line()
    
    def on_click(self, x, y):
        """화면 클릭 콜백"""

        if self.lock:
            return
        
        # 첫번째 점이 없다면 설정한다
        if self.start_point is None:  
            self.start_point = (x, y)
        
        # 있다면 두번째 점을 설정한다
        else:  
            self.end_point = (x, y)

            # 이제 이 클래스는 수정이 불가능하다 - 화면 상에서 고정된다
            self.lock = True
            self.input_manager.unsubscribe(SubscriptionType.LEFT_CLICK, self.subscription_id)

    def render_line(self):
        """
        직선을 렌더링한다
        """

        # 둘 다 입력이 되지 않았거나 (None) 또는 두 점이 일치하는 경우
        if self.start_point == self.end_point:
            return
        
        # 클릭된 두 점으로 이루어진 선분을 연장하여 화면의 가장자리와의 교점을 반환
        extended_ends = self.extend_line()
        self.renderer.push_task(RenderTasks.DRAW_LINE, extended_ends)

    def extend_line(self):
        """클릭된 두 점으로 이루어진 선분을 연장하여 가장자리와의 교점을 반환한다"""

        # 이 작업은 LienarGraph라는 클래스를 이용하여 처리한다
        self.linear_graph = geometrics.LinearGraph(self.start_point, self.end_point)

        return self.linear_graph.boundary_intercepts()
    
class LineTracerManager:
    """
    여러개의 LineTracer를 관리하기 위한 클래스
    """

    def __init__(self, renderer: ImageRenderer, input_manager: InputManager):
        self.renderer = renderer
        self.input_manager = input_manager
        self.old_tracers = []
        self.current_tracer = LineTracer(renderer, input_manager)

    def initialize(self):
        """첫번째 LineTracer를 초기화 시킨다"""

        self.current_tracer.initialize()

    def handle_inputs(self):
        """호출된 이벤트를 처리한다"""

        input_ev = self.input_manager.current_event
        
        # 실행취소 - 마지막 LineTracer를 제거한다
        if input_ev == HairSEMEvents.REMOVE_PREVIOUS_LINE:
            self.revert_last()

        # 분석 시작
        elif input_ev == HairSEMEvents.ANALYZE:

            # LineTracer 적어도 하나는 있어야 한다
            if len(self.old_tracers) == 0:
                raise ValueError("No tracers registered")
            
            # S.SE, 군집 수, 픽셀 수, 표준편차를 출력
            print("running the program")
            sum, n_chunks, n_pixels, std_dev = ml_model.analyze_original_image(self.old_tracers[-1].linear_graph.perpendicular_gradient(), self.renderer.raw_image.copy())
            print(sum, n_chunks, n_pixels, std_dev)

    def cleanup(self):
        """현재 클래스를 초기상태로 초기화"""

        self.old_tracers = []
        self.current_tracer = LineTracer(self.renderer, self.input_manager)

    def update(self):
        """업데이트(tick)"""

        self.handle_inputs()
        
        # 모든 완료 트레이서 업데이트
        for tracer in self.old_tracers:
            tracer.update()

        # 현재 트레이서 업데이트
        self.current_tracer.update()

        # 현재 트레이서가 완료되면, 현재 트레이서를 완료 트레이서 리스트에 추가, 새로운 트레이서 생성
        if self.current_tracer.lock:
            self.old_tracers.append(self.current_tracer)
            self.current_tracer = LineTracer(self.renderer, self.input_manager)
            self.current_tracer.initialize()

    def revert_last(self):
        """가장 최근에 완료된 트레이서 제거"""
        
        if len(self.old_tracers) > 0:
            self.old_tracers.pop()