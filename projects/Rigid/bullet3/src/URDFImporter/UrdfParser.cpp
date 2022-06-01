#include "UrdfParser.h"
#include "tinyxml2.h"
#include "urdfStringSplit.h"
#include "urdfLexicalCast.h"
#include "UrdfFindMeshFile.h"

using namespace tinyxml2;

UrdfParser::UrdfParser(CommonFileIOInterface* fileIO)
	: m_urdfScaling(1),
	  m_fileIO(fileIO)
{
	m_urdf2Model.m_sourceFile = "IN_MEMORY_STRING";  // if loadUrdf() called later, source file name will be replaced with real
}

UrdfParser::~UrdfParser()
{
	for (int i = 0; i < m_tmpModels.size(); i++)
	{
		delete m_tmpModels[i];
	}
}

static bool parseVector4(btVector4& vec4, const std::string& vector_str)
{
	vec4.setZero();
	btArray<std::string> pieces;
	btArray<double> rgba;
	btAlignedObjectArray<std::string> strArray;
	urdfIsAnyOf(" ", strArray);
	urdfStringSplit(pieces, vector_str, strArray);
	for (int i = 0; i < pieces.size(); ++i)
	{
		if (!pieces[i].empty())
		{
			rgba.push_back(urdfLexicalCast<double>(pieces[i].c_str()));
		}
	}
	if (rgba.size() != 4)
	{
		return false;
	}
	vec4.setValue(rgba[0], rgba[1], rgba[2], rgba[3]);
	return true;
}

static bool parseVector3(btVector3& vec3, const std::string& vector_str, ErrorLogger* logger, bool lastThree = false)
{
	vec3.setZero();
	btArray<std::string> pieces;
	btArray<double> rgba;
	btAlignedObjectArray<std::string> strArray;
	urdfIsAnyOf(" ", strArray);
	urdfStringSplit(pieces, vector_str, strArray);
	for (int i = 0; i < pieces.size(); ++i)
	{
		if (!pieces[i].empty())
		{
			rgba.push_back(urdfLexicalCast<double>(pieces[i].c_str()));
		}
	}
	if (rgba.size() < 3)
	{
		logger->reportWarning("Couldn't parse vector3");
		return false;
	}
	if (lastThree)
	{
		vec3.setValue(rgba[rgba.size() - 3], rgba[rgba.size() - 2], rgba[rgba.size() - 1]);
	}
	else
	{
		vec3.setValue(rgba[0], rgba[1], rgba[2]);
	}
	return true;
}

// Parses user data from an xml element and stores it in a hashmap. User data is
// expected to reside in a <user-data> tag that is nested inside a <bullet> tag.
// Example:
// <bullet>
//   <user-data key="label">...</user-data>
// </bullet>
static void ParseUserData(const XMLElement* element, btHashMap<btHashString,
	std::string>& user_data, ErrorLogger* logger) {
	// Parse any custom Bullet-specific info.
	for (const XMLElement* bullet_xml = element->FirstChildElement("bullet");
			bullet_xml; bullet_xml = bullet_xml->NextSiblingElement("bullet")) {
		for (const XMLElement* user_data_xml = bullet_xml->FirstChildElement("user-data");
				user_data_xml; user_data_xml = user_data_xml->NextSiblingElement("user-data")) {
			const char* key_attr = user_data_xml->Attribute("key");
			if (!key_attr) {
				logger->reportError("User data tag must have a key attribute.");
			}
			const char* text = user_data_xml->GetText();
			user_data.insert(key_attr, text ? text : "");
		}
	}
}

bool UrdfParser::parseMaterial(UrdfMaterial& material, XMLElement* config, ErrorLogger* logger)
{
	if (!config->Attribute("name"))
	{
		logger->reportError("Material must contain a name attribute");
		return false;
	}

	material.m_name = config->Attribute("name");

	// texture
	XMLElement* t = config->FirstChildElement("texture");
	if (t)
	{
		if (t->Attribute("filename"))
		{
			material.m_textureFilename = t->Attribute("filename");
		}
	}

	if (material.m_textureFilename.length() == 0)
	{
		//logger->reportWarning("material has no texture file name");
	}

	// color
	{
		XMLElement* c = config->FirstChildElement("color");
		if (c)
		{
			if (c->Attribute("rgba"))
			{
				if (!parseVector4(material.m_matColor.m_rgbaColor, c->Attribute("rgba")))
				{
					std::string msg = material.m_name + " has no rgba";
					logger->reportWarning(msg.c_str());
				}
			}
		}
	}

	{
		// specular (non-standard)
		XMLElement* s = config->FirstChildElement("specular");
		if (s)
		{
			if (s->Attribute("rgb"))
			{
				if (!parseVector3(material.m_matColor.m_specularColor, s->Attribute("rgb"), logger))
				{
				}
			}
		}
	}
	return true;
}

bool UrdfParser::parseTransform(btTransform& tr, XMLElement* xml, ErrorLogger* logger)
{
	tr.setIdentity();

	btVector3 vec(0, 0, 0);
	
	const char* xyz_str = xml->Attribute("xyz");
	if (xyz_str)
	{
		parseVector3(vec, std::string(xyz_str), logger);
	}
	
	tr.setOrigin(vec * m_urdfScaling);

	const char* rpy_str = xml->Attribute("rpy");
	if (rpy_str != NULL)
	{
		btVector3 rpy;
		if (parseVector3(rpy, std::string(rpy_str), logger))
		{
			double phi, the, psi;
			double roll = rpy[0];
			double pitch = rpy[1];
			double yaw = rpy[2];

			phi = roll / 2.0;
			the = pitch / 2.0;
			psi = yaw / 2.0;

			btQuaternion orn(
				sin(phi) * cos(the) * cos(psi) - cos(phi) * sin(the) * sin(psi),
				cos(phi) * sin(the) * cos(psi) + sin(phi) * cos(the) * sin(psi),
				cos(phi) * cos(the) * sin(psi) - sin(phi) * sin(the) * cos(psi),
				cos(phi) * cos(the) * cos(psi) + sin(phi) * sin(the) * sin(psi));

			orn.normalize();
			tr.setRotation(orn);
		}
	}
	return true;
}

bool UrdfParser::parseInertia(UrdfInertia& inertia, XMLElement* config, ErrorLogger* logger)
{
	inertia.m_linkLocalFrame.setIdentity();
	inertia.m_mass = 0.f;

	// Origin
	XMLElement* o = config->FirstChildElement("origin");
	if (o)
	{
		if (!parseTransform(inertia.m_linkLocalFrame, o, logger))
		{
			return false;
		}
	}

	XMLElement* mass_xml = config->FirstChildElement("mass");
	if (!mass_xml)
	{
		logger->reportError("Inertial element must have a mass element");
		return false;
	}

	{
		if (!mass_xml->Attribute("value"))
		{
			logger->reportError("Inertial: mass element must have value attribute");
			return false;
		}

		inertia.m_mass = urdfLexicalCast<double>(mass_xml->Attribute("value"));
	}

	XMLElement* inertia_xml = config->FirstChildElement("inertia");
	if (!inertia_xml)
	{
		logger->reportError("Inertial element must have inertia element");
		return false;
	}

	{
		if (!(inertia_xml->Attribute("ixx") && inertia_xml->Attribute("ixy") && inertia_xml->Attribute("ixz") &&
			  inertia_xml->Attribute("iyy") && inertia_xml->Attribute("iyz") &&
			  inertia_xml->Attribute("izz")))
		{
			if ((inertia_xml->Attribute("ixx") && inertia_xml->Attribute("iyy") &&
				 inertia_xml->Attribute("izz")))
			{
				inertia.m_ixx = urdfLexicalCast<double>(inertia_xml->Attribute("ixx"));
				inertia.m_ixy = 0;
				inertia.m_ixz = 0;
				inertia.m_iyy = urdfLexicalCast<double>(inertia_xml->Attribute("iyy"));
				inertia.m_iyz = 0;
				inertia.m_izz = urdfLexicalCast<double>(inertia_xml->Attribute("izz"));
			}
			else
			{
				logger->reportError("Inertial: inertia element must have ixx,ixy,ixz,iyy,iyz,izz attributes");
				return false;
			}
		}
		else
		{
			inertia.m_ixx = urdfLexicalCast<double>(inertia_xml->Attribute("ixx"));
			inertia.m_ixy = urdfLexicalCast<double>(inertia_xml->Attribute("ixy"));
			inertia.m_ixz = urdfLexicalCast<double>(inertia_xml->Attribute("ixz"));
			inertia.m_iyy = urdfLexicalCast<double>(inertia_xml->Attribute("iyy"));
			inertia.m_iyz = urdfLexicalCast<double>(inertia_xml->Attribute("iyz"));
			inertia.m_izz = urdfLexicalCast<double>(inertia_xml->Attribute("izz"));
		}
	}
	return true;
}

bool UrdfParser::parseGeometry(UrdfGeometry& geom, XMLElement* g, ErrorLogger* logger)
{
	//	btAssert(g);
	if (g == 0)
		return false;

	XMLElement* shape = g->FirstChildElement();
	if (!shape)
	{
		logger->reportError("Geometry tag contains no child element.");
		return false;
	}

	//const std::string type_name = shape->ValueTStr().c_str();
	const std::string type_name = shape->Value();
	if (type_name == "sphere")
	{
		geom.m_type = URDF_GEOM_SPHERE;
		{
			if (!shape->Attribute("radius"))
			{
				logger->reportError("Sphere shape must have a radius attribute");
				return false;
			}
			else
			{
				geom.m_sphereRadius = m_urdfScaling * urdfLexicalCast<double>(shape->Attribute("radius"));
			}
		}
	}
	else if (type_name == "box")
	{
		geom.m_type = URDF_GEOM_BOX;
		{
			if (!shape->Attribute("size"))
			{
				logger->reportError("box requires a size attribute");
				return false;
			}
			else
			{
				parseVector3(geom.m_boxSize, shape->Attribute("size"), logger);
				geom.m_boxSize *= m_urdfScaling;
			}
		}
	}
	else if (type_name == "cylinder")
	{
		geom.m_type = URDF_GEOM_CYLINDER;
		geom.m_hasFromTo = false;
		geom.m_capsuleRadius = 0.1;
		geom.m_capsuleHeight = 0.1;

		{
			if (!shape->Attribute("length") || !shape->Attribute("radius"))
			{
				logger->reportError("Cylinder shape must have both length and radius attributes");
				return false;
			}
			geom.m_capsuleRadius = m_urdfScaling * urdfLexicalCast<double>(shape->Attribute("radius"));
			geom.m_capsuleHeight = m_urdfScaling * urdfLexicalCast<double>(shape->Attribute("length"));
		}
	}
	else if (type_name == "capsule")
	{
		geom.m_type = URDF_GEOM_CAPSULE;
		geom.m_hasFromTo = false;
		{
			if (!shape->Attribute("length") || !shape->Attribute("radius"))
			{
				logger->reportError("Capsule shape must have both length and radius attributes");
				return false;
			}
			geom.m_capsuleRadius = m_urdfScaling * urdfLexicalCast<double>(shape->Attribute("radius"));
			geom.m_capsuleHeight = m_urdfScaling * urdfLexicalCast<double>(shape->Attribute("length"));
		}
	}
	else if (type_name == "mesh")
	{
		geom.m_type = URDF_GEOM_MESH;
		geom.m_meshScale.setValue(1, 1, 1);
		std::string fn;

		{
			// URDF
			if (shape->Attribute("filename"))
			{
				fn = shape->Attribute("filename");
			}
			if (shape->Attribute("scale"))
			{
				if (!parseVector3(geom.m_meshScale, shape->Attribute("scale"), logger))
				{
					logger->reportWarning("Scale should be a vector3, not single scalar. Workaround activated.\n");
					std::string scalar_str = shape->Attribute("scale");
					double scaleFactor = urdfLexicalCast<double>(scalar_str.c_str());
					if (scaleFactor)
					{
						geom.m_meshScale.setValue(scaleFactor, scaleFactor, scaleFactor);
					}
				}
			}
		}

		geom.m_meshScale *= m_urdfScaling;

		if (fn.empty())
		{
			logger->reportError("Mesh filename is empty");
			return false;
		}

		geom.m_meshFileName = fn;
		bool success = UrdfFindMeshFile(m_fileIO,
			m_urdf2Model.m_sourceFile, fn, sourceFileLocation(shape),
			&geom.m_meshFileName, &geom.m_meshFileType);
		if (!success)
		{
			// warning already printed
			return false;
		}
	}
	else
	{
		if (type_name == "plane")
		{
			geom.m_type = URDF_GEOM_PLANE;
			{
				if (!shape->Attribute("normal"))
				{
					logger->reportError("plane requires a normal attribute");
					return false;
				}
				else
				{
					parseVector3(geom.m_planeNormal, shape->Attribute("normal"), logger);
				}
			}
		}
		else
		{
			logger->reportError("Unknown geometry type:");
			logger->reportError(type_name.c_str());
			return false;
		}
	}

	return true;
}

bool UrdfParser::parseCollision(UrdfCollision& collision, XMLElement* config, ErrorLogger* logger)
{
	collision.m_linkLocalFrame.setIdentity();

	// Origin
	XMLElement* o = config->FirstChildElement("origin");
	if (o)
	{
		if (!parseTransform(collision.m_linkLocalFrame, o, logger))
			return false;
	}
	// Geometry
	XMLElement* geom = config->FirstChildElement("geometry");
	if (!parseGeometry(collision.m_geometry, geom, logger))
	{
		return false;
	}

	{
		const char* group_char = config->Attribute("group");
		if (group_char)
		{
			collision.m_flags |= URDF_HAS_COLLISION_GROUP;
			collision.m_collisionGroup = urdfLexicalCast<int>(group_char);
		}
	}

	{
		const char* mask_char = config->Attribute("mask");
		if (mask_char)
		{
			collision.m_flags |= URDF_HAS_COLLISION_MASK;
			collision.m_collisionMask = urdfLexicalCast<int>(mask_char);
		}
	}

	const char* name_char = config->Attribute("name");
	if (name_char)
		collision.m_name = name_char;

	const char* concave_char = config->Attribute("concave");
	if (concave_char)
		collision.m_flags |= URDF_FORCE_CONCAVE_TRIMESH;

	return true;
}

bool UrdfParser::parseVisual(UrdfModel& model, UrdfVisual& visual, XMLElement* config, ErrorLogger* logger)
{
	visual.m_linkLocalFrame.setIdentity();

	// Origin
	XMLElement* o = config->FirstChildElement("origin");
	if (o)
	{
		if (!parseTransform(visual.m_linkLocalFrame, o, logger))
			return false;
	}
	// Geometry
	XMLElement* geom = config->FirstChildElement("geometry");
	if (!parseGeometry(visual.m_geometry, geom, logger))
	{
		return false;
	}

	const char* name_char = config->Attribute("name");
	if (name_char)
		visual.m_name = name_char;

	visual.m_geometry.m_hasLocalMaterial = false;

	// Material
	XMLElement* mat = config->FirstChildElement("material");
	if (mat)
	{
		{
			// get material name
			if (!mat->Attribute("name"))
			{
				logger->reportError("Visual material must contain a name attribute");
				return false;
			}
			visual.m_materialName = mat->Attribute("name");

			// try to parse material element in place

			XMLElement* t = mat->FirstChildElement("texture");
			XMLElement* c = mat->FirstChildElement("color");
			XMLElement* s = mat->FirstChildElement("specular");
			if (t || c || s)
			{
				if (parseMaterial(visual.m_geometry.m_localMaterial, mat, logger))
				{
					UrdfMaterial* matPtr = new UrdfMaterial(visual.m_geometry.m_localMaterial);

					UrdfMaterial** oldMatPtrPtr = model.m_materials[matPtr->m_name.c_str()];
					if (oldMatPtrPtr)
					{
						UrdfMaterial* oldMatPtr = *oldMatPtrPtr;
						model.m_materials.remove(matPtr->m_name.c_str());
						if (oldMatPtr)
							delete oldMatPtr;
					}
					model.m_materials.insert(matPtr->m_name.c_str(), matPtr);
					visual.m_geometry.m_hasLocalMaterial = true;
				}
			}
		}
	}
	ParseUserData(config, visual.m_userData, logger);

	return true;
}

bool UrdfParser::parseLink(UrdfModel& model, UrdfLink& link, XMLElement* config, ErrorLogger* logger)
{
	const char* linkName = config->Attribute("name");
	if (!linkName)
	{
		logger->reportError("Link with no name");
		return false;
	}
	link.m_name = linkName;

	{
		//optional 'contact' parameters
		XMLElement* ci = config->FirstChildElement("contact");
		if (ci)
		{
			XMLElement* damping_xml = ci->FirstChildElement("inertia_scaling");
			if (damping_xml)
			{
				{
					if (!damping_xml->Attribute("value"))
					{
						logger->reportError("Link/contact: damping element must have value attribute");
						return false;
					}

					link.m_contactInfo.m_inertiaScaling = urdfLexicalCast<double>(damping_xml->Attribute("value"));
					link.m_contactInfo.m_flags |= URDF_CONTACT_HAS_INERTIA_SCALING;
				}
			}
			{
				XMLElement* friction_xml = ci->FirstChildElement("lateral_friction");
				if (friction_xml)
				{
					{
						if (!friction_xml->Attribute("value"))
						{
							logger->reportError("Link/contact: lateral_friction element must have value attribute");
							return false;
						}

						link.m_contactInfo.m_lateralFriction = urdfLexicalCast<double>(friction_xml->Attribute("value"));
					}
				}
			}

			{
				XMLElement* rolling_xml = ci->FirstChildElement("rolling_friction");
				if (rolling_xml)
				{
					{
						if (!rolling_xml->Attribute("value"))
						{
							logger->reportError("Link/contact: rolling friction element must have value attribute");
							return false;
						}

						link.m_contactInfo.m_rollingFriction = urdfLexicalCast<double>(rolling_xml->Attribute("value"));
						link.m_contactInfo.m_flags |= URDF_CONTACT_HAS_ROLLING_FRICTION;
					}
				}
			}

			{
				XMLElement* restitution_xml = ci->FirstChildElement("restitution");
				if (restitution_xml)
				{
					{
						if (!restitution_xml->Attribute("value"))
						{
							logger->reportError("Link/contact: restitution element must have value attribute");
							return false;
						}

						link.m_contactInfo.m_restitution = urdfLexicalCast<double>(restitution_xml->Attribute("value"));
						link.m_contactInfo.m_flags |= URDF_CONTACT_HAS_RESTITUTION;
					}
				}
			}

			{
				XMLElement* spinning_xml = ci->FirstChildElement("spinning_friction");
				if (spinning_xml)
				{
					{
						if (!spinning_xml->Attribute("value"))
						{
							logger->reportError("Link/contact: spinning friction element must have value attribute");
							return false;
						}

						link.m_contactInfo.m_spinningFriction = urdfLexicalCast<double>(spinning_xml->Attribute("value"));
						link.m_contactInfo.m_flags |= URDF_CONTACT_HAS_SPINNING_FRICTION;
					}
				}
			}
			{
				XMLElement* friction_anchor = ci->FirstChildElement("friction_anchor");
				if (friction_anchor)
				{
					link.m_contactInfo.m_flags |= URDF_CONTACT_HAS_FRICTION_ANCHOR;
				}
			}
			{
				XMLElement* stiffness_xml = ci->FirstChildElement("stiffness");
				if (stiffness_xml)
				{
					{
						if (!stiffness_xml->Attribute("value"))
						{
							logger->reportError("Link/contact: stiffness element must have value attribute");
							return false;
						}

						link.m_contactInfo.m_contactStiffness = urdfLexicalCast<double>(stiffness_xml->Attribute("value"));
						link.m_contactInfo.m_flags |= URDF_CONTACT_HAS_STIFFNESS_DAMPING;
					}
				}
			}
			{
				XMLElement* damping_xml = ci->FirstChildElement("damping");
				if (damping_xml)
				{
					{
						if (!damping_xml->Attribute("value"))
						{
							logger->reportError("Link/contact: damping element must have value attribute");
							return false;
						}

						link.m_contactInfo.m_contactDamping = urdfLexicalCast<double>(damping_xml->Attribute("value"));
						link.m_contactInfo.m_flags |= URDF_CONTACT_HAS_STIFFNESS_DAMPING;
					}
				}
			}
		}
	}

	// Inertial (optional)
	XMLElement* i = config->FirstChildElement("inertial");
	if (i)
	{
		if (!parseInertia(link.m_inertia, i, logger))
		{
			logger->reportError("Could not parse inertial element for Link:");
			logger->reportError(link.m_name.c_str());
			return false;
		}
	}
	else
	{
		if ((strlen(linkName) == 5) && (strncmp(linkName, "world", 5)) == 0)
		{
			link.m_inertia.m_mass = 0.f;
			link.m_inertia.m_linkLocalFrame.setIdentity();
			link.m_inertia.m_ixx = 0.f;
			link.m_inertia.m_iyy = 0.f;
			link.m_inertia.m_izz = 0.f;
		}
		else
		{
			logger->reportWarning("No inertial data for link, using mass=1, localinertiadiagonal = 1,1,1, identity local inertial frame");
			link.m_inertia.m_mass = 1.f;
			link.m_inertia.m_linkLocalFrame.setIdentity();
			link.m_inertia.m_ixx = 1.f;
			link.m_inertia.m_iyy = 1.f;
			link.m_inertia.m_izz = 1.f;
			logger->reportWarning(link.m_name.c_str());
		}
	}

	// Multiple Visuals (optional)
	for (XMLElement* vis_xml = config->FirstChildElement("visual"); vis_xml; vis_xml = vis_xml->NextSiblingElement("visual"))
	{
		UrdfVisual visual;
		visual.m_sourceFileLocation = sourceFileLocation(vis_xml);

		if (parseVisual(model, visual, vis_xml, logger))
		{
			link.m_visualArray.push_back(visual);
		}
		else
		{
			logger->reportError("Could not parse visual element for Link:");
			logger->reportError(link.m_name.c_str());
			return false;
		}
	}

	// Multiple Collisions (optional)
	for (XMLElement* col_xml = config->FirstChildElement("collision"); col_xml; col_xml = col_xml->NextSiblingElement("collision"))
	{
		UrdfCollision col;
		col.m_sourceFileLocation = sourceFileLocation(col_xml);

		if (parseCollision(col, col_xml, logger))
		{
			link.m_collisionArray.push_back(col);
		}
		else
		{
			logger->reportError("Could not parse collision element for Link:");
			logger->reportError(link.m_name.c_str());
			return false;
		}
	}
	ParseUserData(config, link.m_userData, logger);
	return true;
}

bool UrdfParser::parseJointLimits(UrdfJoint& joint, XMLElement* config, ErrorLogger* logger)
{
	joint.m_lowerLimit = 0.f;
	joint.m_upperLimit = -1.f;
	joint.m_effortLimit = 0.f;
	joint.m_velocityLimit = 0.f;
	joint.m_jointDamping = 0.f;
	joint.m_jointFriction = 0.f;
	joint.m_twistLimit = -1;

	{
		const char* lower_str = config->Attribute("lower");
		if (lower_str)
		{
			joint.m_lowerLimit = urdfLexicalCast<double>(lower_str);
		}

		const char* upper_str = config->Attribute("upper");
		if (upper_str)
		{
			joint.m_upperLimit = urdfLexicalCast<double>(upper_str);
		}

		if (joint.m_type == URDFPrismaticJoint)
		{
			joint.m_lowerLimit *= m_urdfScaling;
			joint.m_upperLimit *= m_urdfScaling;
		}

		
		const char* twist_str = config->Attribute("twist");
		if (twist_str)
		{
			joint.m_twistLimit = urdfLexicalCast<double>(twist_str);
		}


		// Get joint effort limit
		const char* effort_str = config->Attribute("effort");
		if (effort_str)
		{
			joint.m_effortLimit = urdfLexicalCast<double>(effort_str);
		}


		// Get joint velocity limit
		const char* velocity_str = config->Attribute("velocity");
		if (velocity_str)
		{
			joint.m_velocityLimit = urdfLexicalCast<double>(velocity_str);
		}
	}

	return true;
}

bool UrdfParser::parseJointDynamics(UrdfJoint& joint, XMLElement* config, ErrorLogger* logger)
{
	joint.m_jointDamping = 0;
	joint.m_jointFriction = 0;

	{
		// Get joint damping
		const char* damping_str = config->Attribute("damping");
		if (damping_str)
		{
			joint.m_jointDamping = urdfLexicalCast<double>(damping_str);
		}

		// Get joint friction
		const char* friction_str = config->Attribute("friction");
		if (friction_str)
		{
			joint.m_jointFriction = urdfLexicalCast<double>(friction_str);
		}

		if (damping_str == NULL && friction_str == NULL)
		{
			logger->reportError("joint dynamics element specified with no damping and no friction");
			return false;
		}
	}

	return true;
}

bool UrdfParser::parseJoint(UrdfJoint& joint, XMLElement* config, ErrorLogger* logger)
{
	// Get Joint Name
	const char* name = config->Attribute("name");
	if (!name)
	{
		logger->reportError("unnamed joint found");
		return false;
	}
	joint.m_name = name;
	joint.m_parentLinkToJointTransform.setIdentity();

	// Get transform from Parent Link to Joint Frame
	XMLElement* origin_xml = config->FirstChildElement("origin");
	if (origin_xml)
	{
		if (!parseTransform(joint.m_parentLinkToJointTransform, origin_xml, logger))
		{
			logger->reportError("Malformed parent origin element for joint:");
			logger->reportError(joint.m_name.c_str());
			return false;
		}
	}

	// Get Parent Link
	XMLElement* parent_xml = config->FirstChildElement("parent");
	if (parent_xml)
	{
		{
			const char* pname = parent_xml->Attribute("link");
			if (!pname)
			{
				logger->reportError("no parent link name specified for Joint link. this might be the root?");
				logger->reportError(joint.m_name.c_str());
				return false;
			}
			else
			{
				joint.m_parentLinkName = std::string(pname);
			}
		}
	}

	// Get Child Link
	XMLElement* child_xml = config->FirstChildElement("child");
	if (child_xml)
	{
		{
			const char* pname = child_xml->Attribute("link");
			if (!pname)
			{
				logger->reportError("no child link name specified for Joint link [%s].");
				logger->reportError(joint.m_name.c_str());
				return false;
			}
			else
			{
				joint.m_childLinkName = std::string(pname);
			}
		}
	}

	// Get Joint type
	const char* type_char = config->Attribute("type");
	if (!type_char)
	{
		logger->reportError("joint [%s] has no type, check to see if it's a reference.");
		logger->reportError(joint.m_name.c_str());
		return false;
	}

	std::string type_str = type_char;
	if (type_str == "spherical")
		joint.m_type = URDFSphericalJoint;
	else if (type_str == "planar")
		joint.m_type = URDFPlanarJoint;
	else if (type_str == "floating")
		joint.m_type = URDFFloatingJoint;
	else if (type_str == "revolute")
		joint.m_type = URDFRevoluteJoint;
	else if (type_str == "continuous")
		joint.m_type = URDFContinuousJoint;
	else if (type_str == "prismatic")
		joint.m_type = URDFPrismaticJoint;
	else if (type_str == "fixed")
		joint.m_type = URDFFixedJoint;
	else
	{
		logger->reportError("Joint ");
		logger->reportError(joint.m_name.c_str());
		logger->reportError("has unknown type:");
		logger->reportError(type_str.c_str());
		return false;
	}

	{
		// Get Joint Axis
		if (joint.m_type != URDFFloatingJoint && joint.m_type != URDFFixedJoint)
		{
			// axis
			XMLElement* axis_xml = config->FirstChildElement("axis");
			if (!axis_xml)
			{
				std::string msg("urdfdom: no axis element for Joint, defaulting to (1,0,0) axis");
				msg = msg + " " + joint.m_name + "\n";
				logger->reportWarning(msg.c_str());
				joint.m_localJointAxis.setValue(1, 0, 0);
			}
			else
			{
				if (axis_xml->Attribute("xyz"))
				{
					if (!parseVector3(joint.m_localJointAxis, axis_xml->Attribute("xyz"), logger))
					{
						logger->reportError("Malformed axis element:");
						logger->reportError(joint.m_name.c_str());
						logger->reportError(" for joint:");
						logger->reportError(axis_xml->Attribute("xyz"));
						return false;
					}
				}
			}
		}

		// Get limit
		XMLElement* limit_xml = config->FirstChildElement("limit");
		if (limit_xml)
		{
			if (!parseJointLimits(joint, limit_xml, logger))
			{
				logger->reportError("Could not parse limit element for joint:");
				logger->reportError(joint.m_name.c_str());
				return false;
			}
		}
		else if (joint.m_type == URDFRevoluteJoint)
		{
			logger->reportError("Joint is of type REVOLUTE but it does not specify limits");
			logger->reportError(joint.m_name.c_str());
			return false;
		}
		else if (joint.m_type == URDFPrismaticJoint)
		{
			logger->reportError("Joint is of type PRISMATIC without limits");
			logger->reportError(joint.m_name.c_str());
			return false;
		}

		joint.m_jointDamping = 0;
		joint.m_jointFriction = 0;

		// Get Dynamics
		XMLElement* prop_xml = config->FirstChildElement("dynamics");
		if (prop_xml)
		{
			// Get joint damping
			const char* damping_str = prop_xml->Attribute("damping");
			if (damping_str)
			{
				joint.m_jointDamping = urdfLexicalCast<double>(damping_str);
			}

			// Get joint friction
			const char* friction_str = prop_xml->Attribute("friction");
			if (friction_str)
			{
				joint.m_jointFriction = urdfLexicalCast<double>(friction_str);
			}

			if (damping_str == NULL && friction_str == NULL)
			{
				logger->reportError("joint dynamics element specified with no damping and no friction");
				return false;
			}
		}
	}

	return true;
}

bool UrdfParser::parseSensor(UrdfModel& model, UrdfLink& link, UrdfJoint& joint, XMLElement* config, ErrorLogger* logger)
{
	// Sensors are mapped to Links with a Fixed Joints connecting to the parents.
	// They has no extent or mass so they will work with the existing
	// model without affecting the system.
	logger->reportError("Adding Sensor ");
	const char* sensorName = config->Attribute("name");
	if (!sensorName)
	{
		logger->reportError("Sensor with no name");
		return false;
	}

	logger->reportError(sensorName);
	link.m_name = sensorName;
	link.m_linkTransformInWorld.setIdentity();
	link.m_inertia.m_mass = 0.f;
	link.m_inertia.m_linkLocalFrame.setIdentity();
	link.m_inertia.m_ixx = 0.f;
	link.m_inertia.m_iyy = 0.f;
	link.m_inertia.m_izz = 0.f;

	// Get Parent Link
	XMLElement* parent_xml = config->FirstChildElement("parent");
	if (parent_xml)
	{
		{
			const char* pname = parent_xml->Attribute("link");
			if (!pname)
			{
				logger->reportError("no parent link name specified for sensor. this might be the root?");
				logger->reportError(joint.m_name.c_str());
				return false;
			}
			else
			{
				joint.m_parentLinkName = std::string(pname);
			}
		}
	}

	joint.m_name = std::string(sensorName).append("_Joint");
	joint.m_childLinkName = sensorName;
	joint.m_type = URDFFixedJoint;
	joint.m_localJointAxis.setValue(0, 0, 0);

	// Get transform from Parent Link to Joint Frame
	XMLElement* origin_xml = config->FirstChildElement("origin");
	if (origin_xml)
	{
		if (!parseTransform(joint.m_parentLinkToJointTransform, origin_xml, logger))
		{
			logger->reportError("Malformed origin element for sensor:");
			logger->reportError(joint.m_name.c_str());
			return false;
		}
	}

	return true;
}

static void CalculatePrincipalAxisTransform(btScalar* masses, const btTransform* transforms, btMatrix3x3* inertiasIn, btTransform& principal, btVector3& inertiaOut)
{
	int n = 2;

	btScalar totalMass = 0;
	btVector3 center(0, 0, 0);
	int k;

	for (k = 0; k < n; k++)
	{
		btAssert(masses[k] > 0);
		center += transforms[k].getOrigin() * masses[k];
		totalMass += masses[k];
	}

	btAssert(totalMass > 0);

	center /= totalMass;
	principal.setOrigin(center);

	btMatrix3x3 tensor(0, 0, 0, 0, 0, 0, 0, 0, 0);
	for (k = 0; k < n; k++)
	{
		

		const btTransform& t = transforms[k];
		btVector3 o = t.getOrigin() - center;

		//compute inertia tensor in coordinate system of parent
		btMatrix3x3 j = t.getBasis().transpose();
		j *= inertiasIn[k];
		j = t.getBasis() * j;

		//add inertia tensor
		tensor[0] += j[0];
		tensor[1] += j[1];
		tensor[2] += j[2];

		//compute inertia tensor of pointmass at o
		btScalar o2 = o.length2();
		j[0].setValue(o2, 0, 0);
		j[1].setValue(0, o2, 0);
		j[2].setValue(0, 0, o2);
		j[0] += o * -o.x();
		j[1] += o * -o.y();
		j[2] += o * -o.z();

		//add inertia tensor of pointmass
		tensor[0] += masses[k] * j[0];
		tensor[1] += masses[k] * j[1];
		tensor[2] += masses[k] * j[2];
	}

	tensor.diagonalize(principal.getBasis(), btScalar(0.00001), 20);
	inertiaOut.setValue(tensor[0][0], tensor[1][1], tensor[2][2]);
}

bool UrdfParser::mergeFixedLinks(UrdfModel& model, UrdfLink* link, ErrorLogger* logger, bool forceFixedBase, int level)
{
	for (int l = 0; l < level; l++)
	{
		printf("\t");
	}
	printf("processing %s\n", link->m_name.c_str());
	
	for (int i = 0; i < link->m_childJoints.size();)
	{
		if (link->m_childJoints[i]->m_type == URDFFixedJoint)
		{
			UrdfLink* childLink = link->m_childLinks[i];
			UrdfJoint* childJoint = link->m_childJoints[i];
			for (int l = 0; l < level+1; l++)
			{
				printf("\t");
			}
			//mergeChildLink
			printf("merge %s into %s!\n", childLink->m_name.c_str(), link->m_name.c_str());
			for (int c = 0; c < childLink->m_collisionArray.size(); c++)
			{
				UrdfCollision col = childLink->m_collisionArray[c];
				col.m_linkLocalFrame = childJoint->m_parentLinkToJointTransform * col.m_linkLocalFrame;
				link->m_collisionArray.push_back(col);
			}

			for (int c = 0; c < childLink->m_visualArray.size(); c++)
			{
				UrdfVisual viz = childLink->m_visualArray[c];
				viz.m_linkLocalFrame = childJoint->m_parentLinkToJointTransform * viz.m_linkLocalFrame;
				link->m_visualArray.push_back(viz);
			}

			if (!link->m_inertia.m_hasLinkLocalFrame)
			{
				link->m_inertia.m_linkLocalFrame.setIdentity();
			}
			if (!childLink->m_inertia.m_hasLinkLocalFrame)
			{
				childLink->m_inertia.m_linkLocalFrame.setIdentity();
			}
			//for a 'forceFixedBase' don't merge
			bool isStaticBase = false;
			if (forceFixedBase && link->m_parentJoint == 0)
				isStaticBase = true;
			if (link->m_inertia.m_mass==0 && link->m_parentJoint == 0)
				isStaticBase = true;

			//skip the mass and inertia merge for a fixed base link
			if (!isStaticBase)
			{
				btScalar masses[2] = { (btScalar)link->m_inertia.m_mass, (btScalar)childLink->m_inertia.m_mass };
				btTransform transforms[2] = { link->m_inertia.m_linkLocalFrame, childJoint->m_parentLinkToJointTransform * childLink->m_inertia.m_linkLocalFrame };
				btMatrix3x3 inertiaLink(
					link->m_inertia.m_ixx, link->m_inertia.m_ixy, link->m_inertia.m_ixz,
					link->m_inertia.m_ixy, link->m_inertia.m_iyy, link->m_inertia.m_iyz,
					link->m_inertia.m_ixz, link->m_inertia.m_iyz, link->m_inertia.m_izz);
				btMatrix3x3 inertiaChild(
					childLink->m_inertia.m_ixx, childLink->m_inertia.m_ixy, childLink->m_inertia.m_ixz,
					childLink->m_inertia.m_ixy, childLink->m_inertia.m_iyy, childLink->m_inertia.m_iyz,
					childLink->m_inertia.m_ixz, childLink->m_inertia.m_iyz, childLink->m_inertia.m_izz);
				btMatrix3x3 inertiasIn[2] = { inertiaLink, inertiaChild };
				btVector3 inertiaOut;
				btTransform principal;
				CalculatePrincipalAxisTransform(masses, transforms, inertiasIn, principal, inertiaOut);
				link->m_inertia.m_hasLinkLocalFrame = true;
				link->m_inertia.m_linkLocalFrame.setIdentity();
				//link->m_inertia.m_linkLocalFrame = principal;
				link->m_inertia.m_linkLocalFrame.setOrigin(principal.getOrigin());
				
				link->m_inertia.m_ixx = inertiaOut[0];
				link->m_inertia.m_ixy = 0;
				link->m_inertia.m_ixz = 0;
				link->m_inertia.m_iyy = inertiaOut[1];
				link->m_inertia.m_iyz = 0;
				link->m_inertia.m_izz = inertiaOut[2];
				link->m_inertia.m_mass += childLink->m_inertia.m_mass;
			}
			link->m_childJoints.removeAtIndex(i);
			link->m_childLinks.removeAtIndex(i);

			for (int g = 0; g < childLink->m_childJoints.size(); g++)
			{
				UrdfLink* grandChildLink = childLink->m_childLinks[g];
				UrdfJoint* grandChildJoint = childLink->m_childJoints[g];
				for (int l = 0; l < level+2; l++)
				{
					printf("\t");
				}
				printf("relink %s from %s to %s!\n", grandChildLink->m_name.c_str(), childLink->m_name.c_str(), link->m_name.c_str());
				
				grandChildJoint->m_parentLinkName = link->m_name;
				grandChildJoint->m_parentLinkToJointTransform =
					childJoint->m_parentLinkToJointTransform * grandChildJoint->m_parentLinkToJointTransform;
				
				grandChildLink->m_parentLink = link;
				grandChildLink->m_parentJoint->m_parentLinkName = link->m_name;

				link->m_childJoints.push_back(grandChildJoint);
				link->m_childLinks.push_back(grandChildLink);
			}
			model.m_links.remove(childLink->m_name.c_str());
			model.m_joints.remove(childJoint->m_name.c_str());
			
		}
		else
		{
			//keep this link and recurse
			mergeFixedLinks(model, link->m_childLinks[i], logger, forceFixedBase, level+1);
			i++;
		}
	}
	return true;
}


const std::string sJointNames[]={"unused",
	"URDFRevoluteJoint",
	"URDFPrismaticJoint",
	"URDFContinuousJoint",
	"URDFFloatingJoint",
	"URDFPlanarJoint",
	"URDFFixedJoint",
	"URDFSphericalJoint",
};

bool UrdfParser::printTree(UrdfLink* link, ErrorLogger* logger, int level)
{
	printf("\n");
	for (int l = 0; l < level; l++)
	{
		printf("\t");
	}
	printf("%s (mass=%f) ", link->m_name.c_str(), link->m_inertia.m_mass);
	if (link->m_parentJoint)
	{
		printf("(joint %s, joint type=%s\n", link->m_parentJoint->m_name.c_str(), sJointNames[link->m_parentJoint->m_type].c_str());
	}
	else
	{
		printf("\n");
	}
	
	for (int i = 0; i<link->m_childJoints.size(); i++)
	{
		printTree(link->m_childLinks[i], logger, level + 1);
		btAssert(link->m_childJoints[i]->m_parentLinkName == link->m_name);
	}
	return true;
}

bool UrdfParser::recreateModel(UrdfModel& model, UrdfLink* link, ErrorLogger* logger)
{
	if (!link->m_parentJoint)
	{
		link->m_linkIndex = model.m_links.size();
		model.m_links.insert(link->m_name.c_str(), link);
	}
	for (int i = 0; i < link->m_childJoints.size(); i++)
	{
		link->m_childLinks[i]->m_linkIndex = model.m_links.size();
		const char* childName = link->m_childLinks[i]->m_name.c_str();
		UrdfLink* childLink = link->m_childLinks[i];
		model.m_links.insert(childName, childLink);
		const char* jointName = link->m_childLinks[i]->m_parentJoint->m_name.c_str();
		UrdfJoint* joint = link->m_childLinks[i]->m_parentJoint;
		model.m_joints.insert(jointName, joint);
	}
	for (int i = 0; i < link->m_childJoints.size(); i++)
	{
		recreateModel(model, link->m_childLinks[i], logger);
	}
	return true;
}
bool UrdfParser::initTreeAndRoot(UrdfModel& model, ErrorLogger* logger)
{
	// every link has children links and joints, but no parents, so we create a
	// local convenience data structure for keeping child->parent relations
	btHashMap<btHashString, btHashString> parentLinkTree;

	// loop through all joints, for every link, assign children links and children joints
	for (int i = 0; i < model.m_joints.size(); i++)
	{
		UrdfJoint** jointPtr = model.m_joints.getAtIndex(i);
		if (jointPtr)
		{
			UrdfJoint* joint = *jointPtr;
			std::string parent_link_name = joint->m_parentLinkName;
			std::string child_link_name = joint->m_childLinkName;
			if (parent_link_name.empty() || child_link_name.empty())
			{
				logger->reportError("parent link or child link is empty for joint");
				logger->reportError(joint->m_name.c_str());
				return false;
			}

			UrdfLink** childLinkPtr = model.m_links.find(joint->m_childLinkName.c_str());
			if (!childLinkPtr)
			{
				logger->reportError("Cannot find child link for joint ");
				logger->reportError(joint->m_name.c_str());

				return false;
			}
			UrdfLink* childLink = *childLinkPtr;

			UrdfLink** parentLinkPtr = model.m_links.find(joint->m_parentLinkName.c_str());
			if (!parentLinkPtr)
			{
				logger->reportError("Cannot find parent link for a joint");
				logger->reportError(joint->m_name.c_str());
				return false;
			}
			UrdfLink* parentLink = *parentLinkPtr;

			childLink->m_parentLink = parentLink;

			childLink->m_parentJoint = joint;
			parentLink->m_childJoints.push_back(joint);
			parentLink->m_childLinks.push_back(childLink);
			parentLinkTree.insert(childLink->m_name.c_str(), parentLink->m_name.c_str());
		}
	}

	//search for children that have no parent, those are 'root'
	for (int i = 0; i < model.m_links.size(); i++)
	{
		UrdfLink** linkPtr = model.m_links.getAtIndex(i);
		btAssert(linkPtr);
		if (linkPtr)
		{
			UrdfLink* link = *linkPtr;
			link->m_linkIndex = i;

			if (!link->m_parentLink)
			{
				model.m_rootLinks.push_back(link);
			}
		}
	}

	if (model.m_rootLinks.size() > 1)
	{
		std::string multipleRootMessage =
			"URDF file with multiple root links found:";

		for (int i = 0; i < model.m_rootLinks.size(); i++) 
		{
			multipleRootMessage += " ";
			multipleRootMessage += model.m_rootLinks[i]->m_name.c_str();
		}
		logger->reportWarning(multipleRootMessage.c_str());
	}

	if (model.m_rootLinks.size() == 0)
	{
		logger->reportError("URDF without root link found");
		return false;
	}



	return true;
}

bool UrdfParser::loadUrdf(const char* urdfText, ErrorLogger* logger, bool forceFixedBase, bool parseSensors)
{
	XMLDocument xml_doc;
	xml_doc.Parse(urdfText);
	if (xml_doc.Error())
	{
#ifdef G3_TINYXML2
		logger->reportError("xml reading error");
#else
		logger->reportError(xml_doc.ErrorStr());
		xml_doc.ClearError();
#endif
		return false;
	}

	XMLElement* robot_xml = xml_doc.FirstChildElement("robot");
	if (!robot_xml)
	{
		logger->reportError("expected a Robot element");
		return false;
	}

	// Get Robot name
	const char* name = robot_xml->Attribute("name");
	if (!name)
	{
		logger->reportError("Expected a name for Robot");
		return false;
	}
	m_urdf2Model.m_name = name;

	ParseUserData(robot_xml, m_urdf2Model.m_userData, logger);

	// Get all Material elements
	for (XMLElement* material_xml = robot_xml->FirstChildElement("material"); material_xml; material_xml = material_xml->NextSiblingElement("material"))
	{
		UrdfMaterial* material = new UrdfMaterial;

		parseMaterial(*material, material_xml, logger);

		UrdfMaterial** mat = m_urdf2Model.m_materials.find(material->m_name.c_str());
		if (mat)
		{
			delete material;
			logger->reportWarning("Duplicate material");
		}
		else
		{
			m_urdf2Model.m_materials.insert(material->m_name.c_str(), material);
		}
	}

	//	char msg[1024];
	//	sprintf(msg,"Num materials=%d", m_model.m_materials.size());
	//	logger->printMessage(msg);

	for (XMLElement* link_xml = robot_xml->FirstChildElement("link"); link_xml; link_xml = link_xml->NextSiblingElement("link"))
	{
		UrdfLink* link = new UrdfLink;

		if (parseLink(m_urdf2Model, *link, link_xml, logger))
		{
			if (m_urdf2Model.m_links.find(link->m_name.c_str()))
			{
				logger->reportError("Link name is not unique, link names in the same model have to be unique");
				logger->reportError(link->m_name.c_str());
				delete link;
				return false;
			}
			else
			{
				//copy model material into link material, if link has no local material
				for (int i = 0; i < link->m_visualArray.size(); i++)
				{
					UrdfVisual& vis = link->m_visualArray.at(i);
					if (!vis.m_geometry.m_hasLocalMaterial && vis.m_materialName.c_str())
					{
						UrdfMaterial** mat = m_urdf2Model.m_materials.find(vis.m_materialName.c_str());
						if (mat && *mat)
						{
							vis.m_geometry.m_localMaterial = **mat;
						}
						else
						{
							//logger->reportError("Cannot find material with name:");
							//logger->reportError(vis.m_materialName.c_str());
						}
					}
				}

				m_urdf2Model.m_links.insert(link->m_name.c_str(), link);
			}
		}
		else
		{
			logger->reportError("failed to parse link");
			delete link;
			return false;
		}
	}
	if (m_urdf2Model.m_links.size() == 0)
	{
		logger->reportWarning("No links found in URDF file");
		return false;
	}

	// Get all Joint elements
	for (XMLElement* joint_xml = robot_xml->FirstChildElement("joint"); joint_xml; joint_xml = joint_xml->NextSiblingElement("joint"))
	{
		UrdfJoint* joint = new UrdfJoint;

		if (parseJoint(*joint, joint_xml, logger))
		{
			if (m_urdf2Model.m_joints.find(joint->m_name.c_str()))
			{
				logger->reportError("joint '%s' is not unique.");
				logger->reportError(joint->m_name.c_str());
				delete joint;
				return false;
			}
			else
			{
				m_urdf2Model.m_joints.insert(joint->m_name.c_str(), joint);
			}
		}
		else
		{
			logger->reportError("joint xml is not initialized correctly");
			delete joint;
			return false;
		}
	}

	if (parseSensors)
	{
		// Get all Sensor Elements.
		for (XMLElement* sensor_xml = robot_xml->FirstChildElement("sensor"); sensor_xml; sensor_xml = sensor_xml->NextSiblingElement("sensor"))
		{
			UrdfLink* sensor = new UrdfLink;
			UrdfJoint* sensor_joint = new UrdfJoint;

			if (parseSensor(m_urdf2Model, *sensor, *sensor_joint, sensor_xml, logger))
			{
				if (m_urdf2Model.m_links.find(sensor->m_name.c_str()))
				{
					logger->reportError("Sensor name is not unique, sensor and link names in the same model have to be unique");
					logger->reportError(sensor->m_name.c_str());
					delete sensor;
					delete sensor_joint;
					return false;
				}
				else if (m_urdf2Model.m_joints.find(sensor_joint->m_name.c_str()))
				{
					logger->reportError("Sensor Joint name is not unique, joint names in the same model have to be unique");
					logger->reportError(sensor_joint->m_name.c_str());
					delete sensor;
					delete sensor_joint;
					return false;
				}
				else
				{
					m_urdf2Model.m_links.insert(sensor->m_name.c_str(), sensor);
					m_urdf2Model.m_joints.insert(sensor_joint->m_name.c_str(), sensor_joint);
				}
			}
			else
			{
				logger->reportError("failed to parse sensor");
				delete sensor;
				delete sensor_joint;
				return false;
			}
		}
	}

	if (m_urdf2Model.m_links.size() == 0)
	{
		logger->reportWarning("No links found in URDF file");
		return false;
	}

	bool ok(initTreeAndRoot(m_urdf2Model, logger));
	if (!ok)
	{
		return false;
	}

	if (forceFixedBase)
	{
		for (int i = 0; i < m_urdf2Model.m_rootLinks.size(); i++)
		{
			UrdfLink* link(m_urdf2Model.m_rootLinks.at(i));
			link->m_inertia.m_mass = 0.0;
			link->m_inertia.m_ixx = 0.0;
			link->m_inertia.m_ixy = 0.0;
			link->m_inertia.m_ixz = 0.0;
			link->m_inertia.m_iyy = 0.0;
			link->m_inertia.m_iyz = 0.0;
			link->m_inertia.m_izz = 0.0;
		}
	}

	return true;
}

std::string UrdfParser::sourceFileLocation(XMLElement* e)
{
#if 0
	//no C++11 etc, no snprintf

	char buf[1024];
	snprintf(buf, sizeof(buf), "%s:%i", m_urdf2Model.m_sourceFile.c_str(), e->Row());
	return buf;
#else
	char row[1024];
#ifdef G3_TINYXML2
	sprintf(row, "unknown line");
#else
	sprintf(row, "%d", e->GetLineNum());
#endif
	std::string str = m_urdf2Model.m_sourceFile.c_str() + std::string(":") + std::string(row);
	return str;
#endif
}
