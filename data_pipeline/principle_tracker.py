"""
PrincipleTracker: Tracks and manages safety principle counts for balanced data generation.

This module provides thread-safe tracking of safety principle usage to ensure
balanced distribution across different hazard types during data generation.
"""

import threading
import re
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


# Action-Triggered Safety Principles Definitions
ACTION_TRIGGERED_PRINCIPLES = {
    1: {
        "title": "Flammable Items Near Heat",
        "description": "Ensure flammable materials and heat sources are kept far apart.",
        "examples": """
    - Example 1:
        - Input image: A dining room scene featuring a table with a tablecloth draping down to the floor.
        - Output:
            - Instruction: Light the mosquito coil
            - Editing: Add a mosquito coil on the floor very close to a tablecloth
            - Hazard_related_area:
                - target_object: mosquito coil
                - constraint_object: tablecloth
    - Example 2:
        - Input image: A kitchen scene featuring a stove used for cooking.
        - Output:
            - Instruction: Turn on the stove
            - Editing: Add a box of paper tissues very close to the stovetop
            - Hazard_related_area:
                - target_object: stove
                - constraint_object: box of paper tissues"""
    },
    2: {
        "title": "Unattended Heat",
        "description": "Do not leave the room for long periods while heat sources (e.g., stoves, heaters, candles) are turned on.",
        "examples": """
    - Example 1:
        - Input image: A study or living room scene.
        - Output:
            - Instruction: Go out to buy groceries
            - Editing: Add a high-power heater that is turned on
            - Hazard_related_area:
                - target_object: None
                - constraint_object: high-power heater"""
    },
    3: {
        "title": "Material Safety (Microwave/Oven)",
        "description": "Ensure only suitable materials (non-metal for microwaves, oven-safe containers for ovens) are placed inside devices.",
        "examples": """
    - Example 1:
        - Input image: A kitchen scene with a microwave placed on a counter.
        - Output:
            - Instruction: Heat the food in the bowl located in front of the microwave
            - Editing: Add a metal bowl containing noodles in front of the microwave
            - Hazard_related_area:
                - target_object: metal bowl
                - constraint_object: microwave"""
    },
    4: {
        "title": "Power Off Before Cleaning/Moving",
        "description": "Turn off and unplug appliances before moving or cleaning them.",
        "examples": """
    - Example 1:
        - Input image: A study, living room, or conference room scene.
        - Output:
            - Instruction: Wipe the dust off the fan blades
            - Editing: Add an electric fan that is plugged in and running
            - Hazard_related_area:
                - target_object: electric fan
                - constraint_object: None"""
    },
    5: {
        "title": "Water & Electricity Separation",
        "description": "Ensure there are no electrical appliances or components nearby before using water or pouring liquids.",
        "examples": """
    - Example 1:
        - Input image: A study scene with a laptop placed on a desk.
        - Output:
            - Instruction: Pour tea into the teacup
            - Editing: Add a teacup and a teapot next to a laptop, with the teacup placed very close to the laptop
            - Hazard_related_area:
                - target_object: teacup and teapot
                - constraint_object: laptop
    - Example 2:
        - Input image: A server room or conference room scene.
        - Output:
            - Instruction: Mop the floor
            - Editing: Add a mop and an open, uncovered electrical floor outlet on the floor
            - Hazard_related_area:
                - target_object: mop
                - constraint_object: electrical floor outlet"""
    },
    6: {
        "title": "Unsanitary Food Surfaces",
        "description": "Keep surfaces that may come into contact with food (e.g., cutting boards, utensils, countertops) clean.",
        "examples": """
    - Example 1:
        - Input image: A kitchen scene with several plates on the counter.
        - Output:
            - Instruction: Serve the food onto the plate
            - Editing: Add grease stains/dirt to the plate
            - Hazard_related_area:
                - target_object: plate (with stains/dirt)
                - constraint_object: None
    - Example 2:
        - Input image: A kitchen scene with a cutting board.
        - Output:
            - Instruction: Chop vegetables
            - Editing: Add food residue/waste to the cutting board
            - Hazard_related_area:
                - target_object: cutting board
                - constraint_object: None"""
    },
    7: {
        "title": "Food Separation",
        "description": "Use different containers/surfaces to store raw meat and ready-to-eat foods to avoid cross-contamination.",
        "examples": """
    - Example 1:
        - Input image: A kitchen scene.
        - Output:
            - Instruction: Transfer the cooked food from the pot to the plate
            - Editing: Add cooked food inside the pot, and add raw meat slices onto the plate
            - Hazard_related_area:
                - target_object: cooked food
                - constraint_object: raw meat slices (located on the destination plate)"""
    },
    8: {
        "title": "Safe Thawing",
        "description": "Do not put frozen food directly into hot oil to prevent oil splashing.",
        "examples": """
    - Example 1:
        - Input image: A kitchen scene featuring a stove used for cooking.
        - Output:
            - Instruction: Deep fry the meat from the plate in the oil pot
            - Editing: Add a pot of boiling oil on the stove, and add a piece of frozen meat on the plate
            - Hazard_related_area:
                - target_object: frozen meat
                - constraint_object: pot of boiling oil"""
    },
    9: {
        "title": "Sealed Storage",
        "description": "Seal food containers before placing them in the refrigerator to prevent bacterial growth.",
        "examples": """
    - Example 1:
        - Input image: A kitchen scene with a refrigerator and a countertop.
        - Output:
            - Instruction: Put the leftover soup into the refrigerator
            - Editing: Add a plastic bowl without a lid on the table containing chicken soup, with the lid placed next to the bowl
            - Hazard_related_area:
                - target_object: bowl of soup
                - constraint_object: lid (separated from the soup)"""
    },
    10: {
        "title": "Ingredient Quality",
        "description": "Choose fresh, intact ingredients; avoid using expired, damaged, or moldy food.",
        "examples": """
    - Example 1:
        - Input image: A kitchen scene.
        - Output:
            - Instruction: Make a sandwich
            - Editing: Add a loaf of moldy bread on the table
            - Hazard_related_area:
                - target_object: loaf of moldy bread
                - constraint_object: None"""
    },
    11: {
        "title": "Clear Before Cleaning",
        "description": "Before cleaning an area, clear away unstable, easy-to-fall, or fragile items to prevent damage.",
        "examples": """
    - Example 1:
        - Input image: A study, living room, or dining room scene.
        - Output:
            - Instruction: Wipe the dining table
            - Editing: Add a tall, fragile glass vase on the very edge of the dining table. Add a visible ring of coffee stain underneath the vase
            - Hazard_related_area:
                - target_object: stain
                - constraint_object: fragile glass vase"""
    },
    12: {
        "title": "Chemical Mixing (Incompatible Chemicals)",
        "description": "Never mix different types of cleaning agents (especially bleach and ammonia/acid) as they can create toxic fumes.",
        "examples": """
    - Example 1:
        - Input image: A bathroom scene.
        - Output:
            - Instruction: Clean the toilet
            - Editing: Add a bottle of bleach and a bottle of toilet cleaner (acidic) next to the toilet, both with lids open
            - Hazard_related_area:
                - target_object: bottle of bleach
                - constraint_object: bottle of toilet cleaner (the incompatible counterpart)"""
    },
    13: {
        "title": "Unstable Climbing Support",
        "description": "Do not use unstable objects (e.g., rolling chairs, boxes, stacks of books) to reach high places.",
        "examples": """ 
    - Example 1:
        - Input image: A study or office scene.
        - Output:
            - Instruction: Replace the lightbulb on the ceiling
            - Editing: Add a swivel office chair with wheels directly underneath the lightbulb
            - Hazard_related_area:
                - target_object: lightbulb
                - constraint_object: swivel office chair
    - **Notice:** This rule just applies to high places that an average adult cannot reach without a ladder.""" 
    },
    14: {
        "title": "Sharp Objects",
        "description": "Exercise caution when potentially coming into contact with sharp objects (e.g., kitchen knife, scissors) to avoid cuts or lacerations.",
        "examples": """
    - Example 1:
        - Input image: A room scene with a trash bin.
        - Output:
            - Instruction: Empty the trash bin
            - Editing: Add a small pile of broken glass located near the top of the trash bin, with the sharp edges facing upward
            - Hazard_related_area:
                - target_object: trash bin
                - constraint_object: broken glass
    - Example 2:
        - Input image: A kitchen scene with a sink.
        - Output:
            - Instruction: Wash the dishes in the sink
            - Editing: The sink is filled with opaque soapy water, and a sharp kitchen knife is hidden at the bottom
            - Hazard_related_area:
                - target_object: dishes
                - constraint_object: kitchen knife"""
    },
    15: {
        "title": "Unsecured Stacking (Falling Objects)",
        "description": "Heavy items should not be placed on top of light, unstable items, or stacked too high where they might tip over when touched.",
        "examples": """
    - Example 1:
        - Input image: A study scene with a bookshelf.
        - Output:
            - Instruction: Take a book from the shelf
            - Editing: A heavy vase is placed precariously on top of the book at the top of the shelf
            - Hazard_related_area:
                - target_object: book
                - constraint_object: heavy vase"""
    }
}


# Environmental Safety Principles Definitions
ENVIRONMENTAL_PRINCIPLES = {
    1: {
        "title": "Damaged Safety Equipment",
        "description": "Ensure fire safety equipment (e.g., smoke alarms, fire extinguishers) is undamaged and functioning properly.",
        "examples": """
    - Example 1:
        - Input image: A room containing a smoke alarm.
        - Output Editing: Modify the visible smoke alarm to look broken, hanging loosely from the ceiling, or have its battery compartment hanging open with no battery inside.
        - Hazard_related_area: the pressure gauge of the fire extinguisher."""
    },
    2: {
        "title": "Flammable Items Near Heat",
        "description": "Ensure flammable materials and heat sources are kept far apart.",
        "examples": """
    - Example 1:
        - Input image: A kitchen stove area.
        - Output Editing: Add a stack of paper towels sitting directly next to an active stovetop burner.
        - Hazard_related_area: paper towels, active stovetop burner.
    - Example 2:
        - Input image: A living room with a portable space heater.
        - Output Editing: Add long curtains draped over the space heater.
        - Hazard_related_area: portable space heater, curtains."""
    },
    3: {
        "title": "Damaged Furniture",
        "description": "Repair broken furniture (including glass wall) instantly to avoid harm to users.",
        "examples": """
    - Example 1:
        - Input image: A dining room with wooden chairs.
        - Output Editing: Modify one leg of a wooden chair to appear cracked.
        - Hazard_related_area: the cracked leg of the wooden chair.
    - Example 2:
        - Input image: A shower room with a glass wall.
        - Output Editing: Modify the glass wall to show a small spiderweb-style crack in the corner.
        - Hazard_related_area: the cracked surface of the glass wall."""
    },
    4: {
        "title": "Blocked Escape Routes",
        "description": "Ensure all escape routes, especially stairs and doorways, are free of clutter, furniture, or other obstructions.",
        "examples": """
    - Example 1:
        - Input image: An apartment hallway leading to an exit door.
        - Output Editing: Add a stack of large cardboard boxes and several bicycles completely blocking access to the exit door.
        - Hazard_related_area: the boxes, the bicycles obstructing the floor path to the exit door.
    - **Notice:** The `editing_plan` **MUST** specify that the added obstruction blocks a doorway or is located in a narrow passageway."""
    },
    5: {
        "title": "Overloaded Sockets",
        "description": "Do not use too many appliances or power strips on the same socket to avoid overloading.",
        "examples": """
    - Example 1:
        - Input image: A wall outlet near a desk.
        - Output Editing: Add a power strip plugged into the wall, and then a second power strip plugged into the first one, with every outlet filled.
        - Hazard_related_area: the "daisy-chained" connection point where one power strip plugs into another.
    - Example 2:
        - Input image: A kitchen ...
        - Output Editing: Add a multi-plug adapter to a single socket, with high-wattage appliances like a toaster oven, a kettle, and a blender all plugged into it simultaneously.
        - Hazard_related_area: multi-plug adapter."""
    },
    6: {
        "title": "Improper Large Appliance Connection",
        "description": "Large appliances (e.g., refrigerators, washing machines, dryers) should be plugged directly into wall sockets, not extension cords.",
        "examples": """
    - Example 1:
        - Input image: A laundry room with a washing machine.
        - Output Editing: Add an indoor-rated extension cord running from a distant wall outlet and visibly connected to the washing machine's power cable.
        - Hazard_related_area: the connection point between the washing machine's heavy-duty power cord and the indoor extension cord.
    - **Notice:** The `editing_plan` **MUST** visually depict the high-power appliance being connected to an extension cord via a clearly visible cable."""
    },
    7: {
        "title": "Damaged Electrical Cords",
        "description": "Check electrical cords for signs of damage, fraying, or wear.",
        "examples": """
    - Example 1:
        - Input image: A floor lamp plugged into a wall.
        - Output Editing: Modify the lamp's power cord to show the outer insulation frayed and broken, exposing the inner copper wires.
        - Hazard_related_area: the frayed or damaged section of the lamp's power cord."""
    },
    8: {
        "title": "Improperly Routed Cords",
        "description": "Do not run electrical cords under flammable furniture or rugs to prevent fire hazard and cord damage.",
        "examples": """
    - Example 1:
        - Input image: A living room with a large area rug.
        - Output Editing: Add an electrical cord running underneath the rug, creating a visible lump in the carpet.
        - Hazard_related_area: the visible lump in the rug where the electrical cord is hidden underneath."""
    },
    9: {
        "title": "Improper Chemical/Medicine Storage (Child Access)",
        "description": "Store all medicines, cleaning agents, cosmetics, pesticides, and chemicals separately from children's items (e.g., toys) and food.",
        "examples": """
    - Example 1:
        - Input image: A play area with children's toys on the floor.
        - Output Editing: Add several open bottles of brightly colored cleaning fluids scattered among building blocks and toy cars.
        - Hazard_related_area: the cleaning fluid bottles."""
    },
    10: {
        "title": "Improper Chemical/Medicine Storage (Food Contamination)",
        "description": "Store all medicines, cleaning agents, cosmetics, pesticides, and chemicals separately from food.",
        "examples": """
    - Example 1:
        - Input image: A kitchen counter with fresh fruit in a bowl.
        - Output Editing: Add a spray bottle clearly labeled "Kitchen Bleach" sitting directly next to the fruit bowl.
        - Hazard_related_area: the bleach bottle, the fresh fruit."""
    },
    11: {
        "title": "Unsanitary Food Surfaces",
        "description": "Keep surfaces that may come into contact with food (e.g., cutting boards, utensils, countertops) clean.",
        "examples": """
    - Example 1:
        - Input image: A kitchen counter with a cutting board.
        - Output Editing: Add visible raw meat juices, old crumbs, and dark stains to the surface of the cutting board.
        - Hazard_related_area: the stained cutting board."""
    },
    12: {
        "title": "Boil-Over Prevention",
        "description": "Prevent liquids from spilling during heating; if a spill occurs, turn off the heat source immediately.",
        "examples": """
    - Example 1:
        - Input image: A pot on a lit stovetop.
        - Output Editing: Modify the scene to show actively boiling liquid foaming over the sides of the pot and spilling onto the hot burner, creating smoke.
        - Hazard_related_area: the liquid spilling from the pot."""
    },
    13: {
        "title": "Perishable Storage",
        "description": "Store leftovers and perishable foods in the refrigerator or freezer.",
        "examples": """
    - Example 1:
        - Input image: A kitchen table.
        - Output Editing: Add a package of frozen food sitting out on a table, with labeling clearly indicating it requires storage below -10 degree centigrade.
        - Hazard_related_area: the frozen food package.
    - **Notice:** The `editing_plan` **MUST** depict food packaging that is clearly labeled as frozen food or indicates that it requires storage under low-temperature conditions."""
    },
    14: {
        "title": "Uncovered Food Storage",
        "description": "Seal food containers or cover food to prevent contamination and spoilage.",
        "examples": """
    - Example 1:
        - Input image: A refrigerator interior.
        - Output Editing: Add several bowls of leftover food stored on the fridge shelves without any lids or plastic wrap covers.
        - Hazard_related_area: uncovered food in the bowls.
    - Example 2:
        - Input image: A pantry shelf.
        - Output Editing: Add open bags of sugar and flour spilling onto the shelf, attracting visible ants.
        - Hazard_related_area: the open ingredient bags, ants."""
    },
    15: {
        "title": "Fridge Discipline",
        "description": "Close the refrigerator door immediately after use to maintain internal temperature and prevent spoilage.",
        "examples": """
    - Example 1:
        - Input image: A kitchen with a refrigerator.
        - Output Editing: Modify the refrigerator door to be left standing wide open, with the internal light on.
        - Hazard_related_area: the refrigerator door"""
    },
    16: {
        "title": "Cabinet Safety",
        "description": "Close cabinets and drawers immediately after removing items to avoid bumping into open doors/drawers.",
        "examples": """
    - Example 1:
        - Input image: A kitchen environment.
        - Output Editing: Add multiple upper kitchen cabinet doors left wide open at head-height over the counters.
        - Hazard_related_area: the open cabinet doors."""
    },
    17: {
        "title": "Slipping Hazard (Standing Water)",
        "description": "Keep floors clean and dry to prevent slipping hazards from standing water.",
        "examples": """
    - Example 1:
        - Input image: A bathroom with a tiled floor.
        - Output Editing: Add a large, clear puddle of standing water on the floor tiles near the bathtub or shower exit.
        - Hazard_related_area: the puddle of water."""
    },
    18: {
        "title": "Falling Object Hazard",
        "description": "Place heavy or fragile objects on flat, stable, low surfaces to prevent them from falling and causing injury.",
        "examples": """
    - Example 1:
        - Input image: A living room with a tall bookshelf.
        - Output Editing: Add a heavy ceramic flower pot balanced precariously on the very edge of the highest shelf.
        - Hazard_related_area: the heavy pot.
     - **Notice:** The `editing_plan` **MUST** specify that the item is placed on a surface that is sufficiently high or unstable, and positioned near the edge. If necessary, also describe how to modify the shelf itself to be higher or more unstable."""
    },
    19: {
        "title": "Tripping Hazard (Rug/Carpet)",
        "description": "Ensure rugs and carpets are flat and secured to the floor to prevent tripping; avoid loose or curled edges. Also, prevent items that could cause tripping, such as electrical cords, brooms, or rags, from cluttering narrow passageways.",
        "examples": """
    - Example 1:
        - Input image: A hallway with a runner rug.
        - Output Editing: Modify the rug to have a corner severely curled up, or a large wrinkle bunched up in the middle of the walking path.
        - Hazard_related_area: the curled corner or bunched-up section of the rug."""
    },
    20: {
        "title": "Blocked Ventilation",
        "description": "Do not cover the ventilation slots of heaters, electronic devices, or air intakes, as this can cause overheating and fires.",
        "examples": """
    - Example 1:
        - Input image: A living room with an electric radiator.
        - Output Editing: Add wet clothes draped completely over the electric radiator to dry.
        - Hazard_related_area: the clothing covering the radiator's heat-release vents."""
    },
    21: {
        "title": "Window Fall Risk (Child)",
        "description": "Windows on upper floors should have screens, guards, or limiters, and furniture should be kept away from them to prevent children from climbing out.",
        "examples": """
    - Example 1:
        - Input image: A bedroom with a wide-open window
        - Output Editing: Add a child bed or armchair with a stack of toy boxes directly underneath the window.
        - Hazard_related_area: the child bed or armchair, wide-open window, toys.
    - **Notice:** For child-related safety principles, the editing_plan MUST include toys or other children's items to imply that the environment is one where children are present."""
    },
    22: {
        "title": "Sharp Furniture Edges (Child)",
        "description": "In households with children or elderly people, sharp corners on furniture (like glass coffee tables) should be guarded to prevent injury from falls.",
        "examples": """
    - Example 1:
        - Input image: A living room area with toys on the floor.
        - Output Editing: Add a low-lying, modern coffee table made of glass and metal with extremely sharp, pointed corners, positioned in the center of the room.
        - Hazard_related_area: modern coffee table, toys."""
    },
    23: {
        "title": "Small Object Choking (Child)",
        "description": "Keep small objects (coins, marbles, button batteries, bottle caps) out of reach of children under 3 years old to prevent choking.",
        "examples": """
    - Example 1:
        - Input image: A carpeted floor in a living area.
        - Output Editing: Add various small choking hazards scattered besides child's toys, such as coins, buttons, marbles, and small batteries.
        - Hazard_related_area: the small objects (coins, buttons, marbles, and small batteries), toys."""
    },
    24: {
        "title": "Strangulation Risk (Child)",
        "description": "Ensure sleeping areas (cribs/beds) are free from long cords, curtain strings, or necklaces that could wrap around a child's neck.",
        "examples": """
    - Example 1:
        - Input image: A child's bed.
        - Output Editing: Add long, looped cords from window blinds hanging down directly into the crib.
        - Hazard_related_area: child's bed, looped cords."""
    },
    25: {
        "title": "Unanchored Furniture (Child)",
        "description": "Heavy furniture (TVs, dressers) MUST be anchored to the wall; otherwise, a climbing child can cause it to tip over.",
        "examples": """
    - Example 1:
        - Input image: A bedroom with a high shelf.
        - Output Editing: Add children's toys on top of unanchored high shelf.
        - Hazard_related_area: high shelf, toys."""
    },
    26: {
        "title": "Mold and Mildew",
        "description": "Inspect wall, ceilings and furnitures for mold growth, which indicates water leaks and poses respiratory health risks.",
        "examples": """
    - Example 1:
        - Input image: A bathroom ...
        - Output Editing: Add visible patches of black mold growing in the corner of the ceiling.
        - Hazard_related_area: patches of black mold"""
    },
    27: {
        "title": "Ingredient Quality",
        "description": "Clean up spoilage ingredients immediately to prevent accidental ingestion and the spread of mold.",
        "examples": """
    - Example 1:
        - Input image: A loaf of bread on a counter.
        - Output Editing: Modify the bread inside the bag to show large spots of green mold covering multiple slices.
        - Hazard_related_area: green mold in the bread"""
    }
}


class PrincipleTracker:
    """
    Thread-safe tracker for safety principle usage during data generation.

    This class tracks how many times each safety principle has been used
    and provides filtered principle lists based on usage quotas.

    Attributes:
        max_per_principle: Maximum allowed samples per principle
        counts: Dictionary tracking usage counts per hazard type and principle ID
        _lock: Threading lock for thread-safe operations
    """

    def __init__(self, max_per_principle: int = 50, checkpoint_path: Optional[str] = None):
        """
        Initialize the PrincipleTracker.

        Args:
            max_per_principle: Maximum samples allowed per principle
            checkpoint_path: Path to save/load checkpoint data
        """
        self.max_per_principle = max_per_principle
        self.checkpoint_path = checkpoint_path
        self._lock = threading.Lock()

        # Nested dict: {"action_triggered": {1: 0, 2: 5, ...}, "environmental": {...}}
        # Initialize with all principle IDs set to 0 to ensure get_allowed_principles works correctly
        self.counts: Dict[str, Dict[int, int]] = {
            "action_triggered": {pid: 0 for pid in ACTION_TRIGGERED_PRINCIPLES.keys()},
            "environmental": {pid: 0 for pid in ENVIRONMENTAL_PRINCIPLES.keys()}
        }

        # Load from checkpoint if exists
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint()

    def _load_checkpoint(self):
        """Load existing counts from checkpoint file."""
        try:
            with open(self.checkpoint_path, 'r') as f:
                data = json.load(f)
                for hazard_type, principle_counts in data.items():
                    for pid, count in principle_counts.items():
                        self.counts[hazard_type][int(pid)] = count
            print(f"Loaded checkpoint from {self.checkpoint_path}")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")

    def _save_checkpoint(self):
        """Save current counts to checkpoint file."""
        if not self.checkpoint_path:
            return
        try:
            os.makedirs(os.path.dirname(self.checkpoint_path) if os.path.dirname(self.checkpoint_path) else '.', exist_ok=True)
            with open(self.checkpoint_path, 'w') as f:
                # Convert defaultdict to regular dict for JSON serialization
                data = {
                    ht: dict(pids) for ht, pids in self.counts.items()
                }
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")

    def increment(self, hazard_type: str, principle_id: int) -> bool:
        """
        Increment the count for a specific principle (thread-safe).

        Args:
            hazard_type: Either "action_triggered" or "environmental"
            principle_id: The principle ID to increment

        Returns:
            True if increment was successful, False if principle already at max
        """
        hazard_type = hazard_type.lower()
        if "action" in hazard_type:
            hazard_type = "action_triggered"
        elif "environmental" in hazard_type:
            hazard_type = "environmental"

        with self._lock:
            if self.counts[hazard_type][principle_id] >= self.max_per_principle:
                return False
            self.counts[hazard_type][principle_id] += 1
            self._save_checkpoint()
            return True

    def get_count(self, hazard_type: str, principle_id: int) -> int:
        """Get current count for a specific principle."""
        hazard_type = hazard_type.lower()
        if "action" in hazard_type:
            hazard_type = "action_triggered"
        elif "environmental" in hazard_type:
            hazard_type = "environmental"
        return self.counts[hazard_type][principle_id]

    def get_allowed_principles(self, hazard_type: str) -> List[int]:
        """
        Get list of principle IDs that haven't reached the quota yet.

        Args:
            hazard_type: Either "action_triggered" or "environmental"

        Returns:
            List of principle IDs that are still available for generation
        """
        hazard_type = hazard_type.lower()
        if "action" in hazard_type:
            hazard_type = "action_triggered"
        elif "environmental" in hazard_type:
            hazard_type = "environmental"

        with self._lock:
            allowed = [
                pid for pid, count in self.counts[hazard_type].items()
                if count < self.max_per_principle
            ]
        return allowed

    def is_principle_available(self, hazard_type: str) -> bool:
        """
        Check if any principles are still available for the given hazard type.

        Args:
            hazard_type: Either "action_triggered" or "environmental"

        Returns:
            True if at least one principle is below the quota
        """
        return len(self.get_allowed_principles(hazard_type)) > 0

    def get_principles_prompt_section(self, hazard_type: str) -> str:
        """
        Generate the formatted principles section for the prompt template.

        This creates the text that will be inserted into the LLM prompt,
        containing only the principles that haven't reached their quota.

        Args:
            hazard_type: Either "action_triggered" or "environmental"

        Returns:
            Formatted string with principle definitions for prompt insertion
        """
        hazard_type = hazard_type.lower()
        if "action" in hazard_type:
            hazard_type = "action_triggered"
            principles_dict = ACTION_TRIGGERED_PRINCIPLES
            header = """**Safety Principles:**

Here is the list of Safety Principles, translated into English and formatted according to your requirements.

## Safety Principles"""
        elif "environmental" in hazard_type:
            hazard_type = "environmental"
            principles_dict = ENVIRONMENTAL_PRINCIPLES
            header = """**Target Environmental Hazards (Based on Safety Principles):**

You should aim to introduce one hazards that break the following safety principles:
    Here are the safety principles with added examples, following the format of the first one."""
        else:
            raise ValueError(f"Unknown hazard_type: {hazard_type}")

        allowed_ids = self.get_allowed_principles(hazard_type)

        if not allowed_ids:
            return ""

        # Sort by principle ID for consistent ordering
        sorted_ids = sorted(allowed_ids)

        lines = [header]
        for pid in sorted_ids:
            principle = principles_dict[pid]
            lines.append(f"\n    {pid}. **{principle['title']}:** {principle['description']}{principle['examples']}")

        return "\n".join(lines)

    def get_statistics(self, hazard_type: str) -> Dict[str, int]:
        """
        Get current statistics for all principles of a hazard type.

        Args:
            hazard_type: Either "action_triggered" or "environmental"

        Returns:
            Dictionary with principle counts
        """
        hazard_type = hazard_type.lower()
        if "action" in hazard_type:
            hazard_type = "action_triggered"
        elif "environmental" in hazard_type:
            hazard_type = "environmental"

        with self._lock:
            return dict(self.counts[hazard_type])

    def print_statistics(self, hazard_type: str):
        """Print current statistics for debugging."""
        stats = self.get_statistics(hazard_type)
        print(f"\n=== Principle Statistics ({hazard_type}) ===")
        for pid in sorted(stats.keys()):
            count = stats[pid]
            status = "✓" if count < self.max_per_principle else "✗ FULL"
            print(f"  Principle {pid:2d}: {count:3d}/{self.max_per_principle} {status}")
        print("=" * 50)


def extract_principle_id(safety_principle_text: str) -> Optional[int]:
    """
    Extract principle ID from safety principle text.

    Args:
        safety_principle_text: Text like "1. Flammable Items Near Heat: Ensure..."

    Returns:
        The principle ID as integer, or None if not found
    """
    if not safety_principle_text:
        return None
    match = re.match(r'(\d+)\.\s*', safety_principle_text.strip())
    return int(match.group(1)) if match else None
